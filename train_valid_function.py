# train_valid_function.py
import torch
from tqdm import tqdm
import numpy as np
from utils.terminal import MetricMonitor


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    model.train()
    metric_monitor = MetricMonitor(float_precision=4)
    stream = tqdm(train_loader)
    loss_history = []
    train_accuracy = []

    for batch_idx, train_batch in enumerate(stream):
        inputs = torch.cat((train_batch["cover"], train_batch["stego"]), 0)
        labels = torch.cat(
            (train_batch["label"][0], train_batch["label"][1]), 0
        )
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(loss.item())
        prediction = outputs.data.max(1)[1]
        accuracy = (
                prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
        )
        train_accuracy.append(accuracy.item())

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("ACC", accuracy)
        stream.set_description(
            "Epoch: {epoch}. Training.   {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )

    train_losses_avg = np.mean(np.array(loss_history))
    train_acc_avg = np.mean(np.array(train_accuracy))
    return train_losses_avg, train_acc_avg


def validate(model, val_loader, loss_fn, device, epoch, optimizer, logger, c, train_losses_avg, train_acc_avg):
    model.eval()
    loss_history = []
    val_accuracy = []
    metric_monitor = MetricMonitor(float_precision=4)
    stream = tqdm(val_loader)

    with torch.no_grad():
        for batch_idx, val_batch in enumerate(stream):
            inputs = torch.cat((val_batch["cover"], val_batch["stego"]), 0)
            labels = torch.cat(
                (val_batch["label"][0], val_batch["label"][1]), 0
            )

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss_history.append(loss.item())
            prediction = outputs.data.max(1)[1]
            accuracy = (
                    prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
            )
            val_accuracy.append(accuracy.item())

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("ACC", accuracy)
            stream.set_description(
                "Epoch: {epoch}. Validating.  {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

    val_losses_avg = np.mean(np.array(loss_history))
    val_acc_avg = np.mean(np.array(val_accuracy))

    # 确保在训练和验证过程之后执行此代码
    lr = optimizer.state_dict()['param_groups'][0]['lr']  # 获取当前学习率
    logger.info(
        'Epoch: {}/{}, Learning Rate: {:.5f} | Training, AVG_Loss: {:.4f}, AVG_ACC: {:.4f} | Validating, AVG_Loss: {:.4f}, AVG_ACC: {:.4f}'.format(
            epoch, c.epochs, lr, train_losses_avg, train_acc_avg, val_losses_avg, val_acc_avg
        )
    )
    return val_losses_avg, val_acc_avg
