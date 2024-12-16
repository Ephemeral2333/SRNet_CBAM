import torch
import os
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from model.SRNet import Model
from utils.logger import logger_info
from dataset.dataset import get_test_loader
import config as c
from utils.terminal import MetricMonitor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create directories for saving results
results_save_dir = os.path.join('results', 'SRNet_Matrix')
os.makedirs(results_save_dir, exist_ok=True)

# Logging setup
logger_name = 'test'
logger_info(logger_name, log_path=os.path.join(results_save_dir, logger_name + '.log'))
logger = logging.getLogger(logger_name)

# Initialize model
model = Model().to(device)

# Load pretrained model
if c.pre_trained_srnet_path is not None:
    model.load_state_dict(torch.load(c.pre_trained_srnet_path))
    logger.info('Loaded pre-trained model from {:s}'.format(c.pre_trained_srnet_path))
    logger.info('Testing dir {:s}'.format(c.test_data_dir))
else:
    logger.error('No pre-trained model path provided.')
    exit()

# Data loader
test_loader = get_test_loader(c.test_data_dir, c.test_batch_size)

# Initialize predictions and labels storage
all_preds = []
all_labels = []

# Testing loop
model.eval()
metric_monitor = MetricMonitor(float_precision=4)
stream = tqdm(test_loader)

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(stream):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs)
        prediction = outputs.data.max(1)[1]

        all_preds.extend(prediction.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        accuracy = (prediction.eq(labels.data).sum() * 100.0 / labels.size()[0])
        metric_monitor.update("ACC", accuracy)
        stream.set_description("Testing.  {metric_monitor}".format(metric_monitor=metric_monitor))

# Calculate precision, recall, and F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
logger.info('Testing, Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}'.format(precision, recall, f1_score))

# 计算混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_preds)
logger.info('Confusion Matrix:\n{}'.format(conf_matrix))

# 从混淆矩阵计算准确度
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
accuracy_from_confusion_matrix = (TP + TN) / (TP + TN + FP + FN)

# 打印混淆矩阵的准确度及其计算公式
logger.info('Accuracy from Confusion Matrix: {:.4f}'.format(accuracy_from_confusion_matrix))
logger.info('Accuracy Calculation: (TP + TN) / (TP + TN + FP + FN)')
logger.info('Where:')
logger.info('TP (True Positives) = Number of correct positive predictions')
logger.info('TN (True Negatives) = Number of correct negative predictions')
logger.info('FP (False Positives) = Number of incorrect positive predictions')
logger.info('FN (False Negatives) = Number of incorrect negative predictions')

# 同时在控制台打印出来
print('Confusion Matrix:\n', conf_matrix)
print('Accuracy from Confusion Matrix: {:.4f}'.format(accuracy_from_confusion_matrix))
print('Accuracy Calculation: (TP + TN) / (TP + TN + FP + FN)')
print('Where:')
print('TP (True Positives) = Number of correct positive predictions: {}'.format(TP))
print('TN (True Negatives) = Number of correct negative predictions: {}'.format(TN))
print('FP (False Positives) = Number of incorrect positive predictions: {}'.format(FP))
print('FN (False Negatives) = Number of incorrect negative predictions: {}'.format(FN))