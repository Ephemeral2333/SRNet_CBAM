# test_model.py
import torch
import os
import logging
import numpy as np
from tqdm import tqdm

from model.SRNet_CBAM import Model
from utils.logger import logger_info
from dataset.dataset import get_test_loader
import config as c
from utils.terminal import MetricMonitor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create directories for saving results
results_save_dir = os.path.join('results', 'SRNet_CBAM_2')
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

# Testing loop
model.eval()
test_accuracy = []
metric_monitor = MetricMonitor(float_precision=4)
stream = tqdm(test_loader)

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(stream):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs)
        prediction = outputs.data.max(1)[1]
        accuracy = (prediction.eq(labels.data).sum() * 100.0 / labels.size()[0])
        test_accuracy.append(accuracy.item())
        metric_monitor.update("ACC", accuracy)
        stream.set_description("Testing.  {metric_monitor}".format(metric_monitor=metric_monitor))

test_acc_avg = np.mean(np.array(test_accuracy))
logger.info('Testing, AVG_ACC: {:.4f}'.format(test_acc_avg))