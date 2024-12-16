from typing import Any
import config as c


def adjust_learning_rate(optimizer: Any, epoch: int) -> None:
    """Sets the learning rate to the initial learning_rate and decays by 10
    every 30 epochs."""
    learning_rate = c.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
