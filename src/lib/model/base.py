"""This is the module that defines the base model.
"""  # noqa: INP001

from collections.abc import Callable
from logging import getLogger
from typing import override

import lightning as L  # noqa: N812
import torch as to
import torchmetrics

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(classes: dict[str, Callable]) -> None:
    """Checks the :class:`BaseModel` parameters.

    Args:
        classes (dict[str, Callable]): class list.
    """
    error = False # error: True
    keys = [K.MODEL, K.OPT, K.LOSS, K.METRICS]
    for key in keys:
        if key not in classes:
            error = True
            LOGGER.error(f'The key "{key}" for variable "classes" is missing.')
    if error:
        raise ValueError


class BaseModel(L.LightningModule):
    """Defines the base model.

    *   You can customize :meth:`training_step` and :meth:`validation_step` using
        ``.fit``.

    Args:
        classes (dict[str, Callable]): class list.
    """
    def __init__(self, classes: dict[str, Callable]) -> None:
        check_params(classes=classes)
        super().__init__()

        self.model = classes[K.MODEL]
        self.opt = classes[K.OPT]
        self.loss = classes[K.LOSS]
        self.train_metrics = classes[K.METRICS]
        if classes[K.METRICS]:
            self.valid_metrics = classes[K.METRICS].clone(prefix='val_')
        else:
            self.valid_metrics = None
        self.train_metrics_loss = torchmetrics.MeanMetric()
        self.valid_metrics_loss = torchmetrics.MeanMetric()

    def update_metrics(self, train: bool, data: tuple[to.Tensor]) -> dict[str, float]:  # noqa: FBT001
        """Updates metrics.

        Args:
            train (bool): train flag (training step: True, validation step: False).
            data (tuple[to.Tensor]): tuple of labels, preds, and losses.

        Returns:
            dict[str, float]: all metrics results.
        """
        labels, preds, losses = data
        if train:
            res = self.train_metrics(preds, labels)
            res['loss'] = self.train_metrics_loss(losses)
        else:
            res = self.valid_metrics(preds, labels)
            res['val_loss'] = self.valid_metrics_loss(losses)
        return res

    @override
    def configure_optimizers(self) -> Callable:
        """Returns the optimizer method class.

        This function is decorated by ``@override``.

        Returns:
            Callable: optimizer method class.
        """
        return self.opt

    @override
    def training_step(
        self,
        batch: tuple[to.Tensor, to.Tensor],
        batch_idx: int,
    ) -> to.Tensor:
        """Trains the model one step at a time.

        This function is decorated by ``@override``.

        #.  Output predictions. (forward propagation)
        #.  Output losses.
        #.  Update metrics.
        #.  Output log.

        Args:
            batch (tuple[to.Tensor]): tuple of inputs and labels.
            batch_idx (int): batch index.

        Returns:
            to.Tensor: loss between the label and the model prediction.
        """
        inputs, labels = batch
        preds = self.model(inputs)
        losses = self.loss(preds, labels)

        res = self.update_metrics(train=True, data=(labels, preds, losses))
        self.log_dict(dictionary=res, prog_bar=True, on_step=False, on_epoch=True)
        return losses

    @override
    def validation_step(
        self,
        batch: tuple[to.Tensor, to.Tensor],
        batch_idx: int,
    ) -> to.Tensor:
        """Validations the model one step at a time.

        This function is decorated by ``@override``.

        #.  Output predictions. (forward propagation)
        #.  Output losses.
        #.  Update metrics.
        #.  Output log.

        Args:
            batch (tuple[to.Tensor]): tuple of inputs and labels.
            batch_idx (int): batch index.

        Returns:
            to.Tensor: loss between the label and the model prediction.
        """
        inputs, labels = batch
        preds = self.model(inputs)
        losses = self.loss(preds, labels)

        res = self.update_metrics(train=False, data=(labels, preds, losses))
        self.log_dict(dictionary=res, prog_bar=True, on_step=False, on_epoch=True)
        return losses
