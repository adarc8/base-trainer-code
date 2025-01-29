import torch


class Metric:
    """This is a generic class on how to define a metric (whether it is a loss function or just a metric to log)
    loss_weight = 0 -> means it will not be used in the loss function (just for logging)"""

    def __init__(self, loss_fn: torch.nn.Module, loss_weight: int | float, move_to_long: bool = False):
        self.loss_weight = loss_weight
        self._loss_fn = loss_fn
        self._move_to_long = move_to_long

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._move_to_long:
            x = x.long()
            y = y.long()
        metric_val = self._loss_fn(x, y)
        return metric_val


class Regularizer:
    """This is a generic class on how to define a metric (whether it is a loss function or just a metric to log)
    loss_weight = 0 -> means it will not be used in the loss function (just for logging)"""

    def __init__(self, loss_weight: int | float):
        self.loss_weight = loss_weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()
