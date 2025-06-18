"""This is the module that sets up metrics.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any

import torchmetrics
import torchmetrics.classification

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any], func: dict[str, Callable]) -> None:
    """Checks the :class:`SetupMetrics` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    for kind in params[K.METRICS][K.KIND]:
        if kind not in func:
            error = True
            LOGGER.error(
                f'SetupMetrics class does not have a method "{kind}" that '
                f'sets the metrics.',
            )
    if error:
        LOGGER.error('The available metrics are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupMetrics:
    """Sets up metrics.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'mse': self.mse,
            'bacc': self.bacc,
            'macc': self.macc,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> Callable:
        """Sets up metrics.

        Returns:
            Callable: metrics class. (``torchmetrics.MetricCollection``)
        """
        metrics = {}
        for kind in self.params[K.METRICS][K.KIND]:
            self._params = self.params[K.METRICS][kind]
            metrics[kind] = self.func[kind]()
        metrics = torchmetrics.MetricCollection(metrics=metrics)
        return metrics

    def mse(self) -> Callable:
        """Sets ``torchmetrics.MeanSquaredError``.

        Returns:
            Callable: metrics class.
        """
        metrics = torchmetrics.MeanSquaredError(
            squared=self._params['squared'],
            num_outputs=self._params['num_outputs'],
        )
        return metrics

    def bacc(self) -> Callable:
        """Sets ``torchmetrics.classification.BinaryAccuracy``.

        Returns:
            Callable: metrics class.
        """
        metrics = torchmetrics.classification.BinaryAccuracy(
            threshold=self._params['threshold'],
            multidim_average=self._params['multidim_average'],
            ignore_index=self._params['ignore_index'],
            validate_args=self._params['validate_args'],
        )
        return metrics

    def macc(self) -> Callable:
        """Sets ``torchmetrics.classification.MulticlassAccuracy``.

        Returns:
            Callable: metrics class.
        """
        metrics = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self._params['num_classes'],
            top_k=self._params['top_k'],
            average=self._params['average'],
            multidim_average=self._params['multidim_average'],
            ignore_index=self._params['ignore_index'],
            validate_args=self._params['validate_args'],
        )
        return metrics
