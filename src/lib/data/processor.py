"""This is the module that process data.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any

import torch as to

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any], func: dict[str, Callable]) -> None:
    """Checks the :class:`Processor` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    for kind in params[K.PROCESS][K.KIND]:
        if kind not in func:
            error = True
            LOGGER.error(
            f'Processor class does not have a method "{kind}" that '
            f'sets the processing method.',
        )
    if error:
        LOGGER.error('The available processing method are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class Processor:
    """Processes data.

    *   Used to process data when making a ``to.utils.data.DataLoader``
        data pipeline.
    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'one_hot': self.one_hot,
            'rescale': self.rescale,
        }
        check_params(params=params, func=self.func)

    def run(self, x: to.Tensor, y: to.Tensor) -> tuple[to.Tensor, to.Tensor]:
        """Runs process data.

        Args:
            x (to.Tensor): input. (before process)
            y (to.Tensor): label. (before process)

        Returns:
            to.Tensor: input. (after process)
            to.Tensor: label. (after process)
        """
        for kind in self.params[K.PROCESS][K.KIND]:
            self._params = self.params[K.PROCESS][kind]
            x, y = self.func[kind](x, y)
        return x, y

    def one_hot(self, x: to.Tensor, y: to.Tensor) -> tuple[to.Tensor, to.Tensor]:
        """Runs ``to.nn.functional.one_hot``.

        Args:
            x (to.Tensor): input. (before process)
            y (to.Tensor): label. (before process)

        Returns:
            to.Tensor: input. (after process)
            to.Tensor: label. (after process)
        """
        y = to.nn.functional.one_hot(
            y.to(dtype=to.int64),
            num_classes=self._params['num_classes'],
        )
        y = to.squeeze(y)
        return x, y

    def rescale(self, x: to.Tensor, y: to.Tensor) -> tuple[to.Tensor, to.Tensor]:
        """Runs rescale and offset.

        Args:
            x (to.Tensor): input. (before process)
            y (to.Tensor): label. (before process)

        Returns:
            to.Tensor: input. (after process)
            to.Tensor: label. (after process)
        """
        x = x * self._params['scale'] + self._params['offset']
        return x, y
