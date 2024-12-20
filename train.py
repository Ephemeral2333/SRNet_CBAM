# train.py
import torch
import argparse
import os
import wandb
import config as c
from model.SRNet_CBAM import Model
from utils.dirs import mkdirs
from utils.logger import logger_info
from dataset.dataset import get_train_loader, get_val_loader
from model.EarlyStopping import EarlyStopping as early_stopping
from train_valid_function import train_one_epoch, validate
import logging

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Using argument parser to get the configuration
parser = argparse.ArgumentParser(description='SRNet_CBAM')
parser.add_argument('--train_data_dir', type=str, default=c.train_data_dir, help='train data dir')
parser.add_argument('--val_data_dir', type=str, default=c.val_data_dir, help='val data dir')
parser.add_argument('--epochs', type=int, default=c.epochs, help='number of epochs')
parser.add_argument('--pretrained_path', type=str, default=c.pre_trained_srnet_path, help='pre-trained model path')


# Create directories for saving models and results
model_save_dir = os.path.join('checkpoints', 'SRNet_CBAM')
results_save_dir = os.path.join('results', 'SRNet_CBAM')
mkdirs(model_save_dir)
mkdirs(results_save_dir)

# Initialize WandB
wandb.init(project="SRNet_CBAM", config={
    "learning_rate": c.lr,
    "epochs": parser.parse_args().epochs,
    "batch_size": c.train_batch_size,
    "weight_decay": c.weight_decay
})

# 日志记录
logger_name = c.mode
logger_info(logger_name, log_path=os.path.join(results_save_dir, logger_name + '.log'))
logger = logging.getLogger(logger_name)
logger.info('#' * 50)
logger.info('mode: {:s}'.format(c.mode))
logger.info('model: SRNet_CBAM')
logger.info('train data dir: {:s}'.format(parser.parse_args().train_data_dir))
logger.info('val data dir: {:s}'.format(parser.parse_args().val_data_dir))

# Initialize model and early stopping
model = Model().to(device)
early_stopping = early_stopping(patience=7, verbose=True)

# Load pretrained model if available
if parser.parse_args().pretrained_path:
    model.load_state_dict(torch.load(parser.parse_args().pretrained_path))
    logger.info('Load pre-trained model from {:s}'.format(parser.parse_args().pretrained_path))

# Data loaders
train_loader = get_train_loader(parser.parse_args().train_data_dir, c.train_batch_size)
val_loader = get_val_loader(parser.parse_args().val_data_dir, c.val_batch_size)

# Loss function, optimizer, and scheduler
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=c.weight_decay_step, gamma=c.gamma)

# Training loop
for epoch in range(parser.parse_args().epochs):
    epoch += 1

    train_losses_avg, train_acc_avg = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
    scheduler.step()

    wandb.log({'epoch': epoch, 'loss_train': train_losses_avg, 'accuracy_train': train_acc_avg})

    if epoch % c.val_freq == 0:
        val_losses_avg, val_acc_avg = validate(model, val_loader, loss_fn, device, epoch, optimizer, logger, c, train_losses_avg,
                                               train_acc_avg)

        wandb.log({'epoch': epoch, 'loss_val': val_losses_avg, 'accuracy_val': val_acc_avg})

        early_stopping(val_losses_avg, model)

    # Checkpointing
    if epoch % c.save_freq == 0 and epoch >= c.start_save_epoch:
        torch.save(model.state_dict(), os.path.join(model_save_dir, 'checkpoint_%.3i' % epoch + '.pt'))

    # Early stopping
    if early_stopping.early_stop:
        logger.info("Early stopping")
        break

wandb.finish()
