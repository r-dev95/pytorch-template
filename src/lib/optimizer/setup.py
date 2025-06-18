"""This is the module that sets up optimizer method.
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
    """Checks the :class:`SetupOpt` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    kind = params[K.OPT][K.KIND]
    if kind not in func:
        error = True
        LOGGER.error(
            f'SetupOpt class does not have a method "{kind}" that '
            f'sets the optimizer method.',
        )
    if error:
        LOGGER.error('The available optimizer method are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupOpt:
    """Sets up optimizer method.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'sgd': self.sgd,
            'adam': self.adam,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> Callable:
        """Sets up optimizer method.

        Returns:
            Callable: optimizer method class.
        """
        kind = self.params[K.OPT][K.KIND]
        self._params = self.params[K.OPT][kind]
        opt = self.func[kind]()
        return opt

    def sgd(self) -> Callable:
        """Sets ``to.optim.SGD``.

        Returns:
            Callable: optimizer method class.
        """
        opt = to.optim.SGD(
            params=self.params[K.MODEL_PARAMS],
            lr=self._params['lr'],
            momentum=self._params['momentum'],
            dampening=self._params['dampening'],
            weight_decay=self._params['weight_decay'],
            nesterov=self._params['nesterov'],
            maximize=self._params['maximize'],
            foreach=self._params['foreach'],
            differentiable=self._params['differentiable'],
            fused=self._params['fused'],
        )
        return opt

    def adam(self) -> Callable:
        """Sets ``to.optim.Adam``.

        Returns:
            Callable: optimizer method class.
        """
        opt = to.optim.Adam(
            params=self.params[K.MODEL_PARAMS],
            lr=self._params['lr'],
            betas=self._params['betas'],
            eps=self._params['eps'],
            weight_decay=self._params['weight_decay'],
            amsgrad=self._params['amsgrad'],
            foreach=self._params['foreach'],
            maximize=self._params['maximize'],
            capturable=self._params['capturable'],
            differentiable=self._params['differentiable'],
            fused=self._params['fused'],
        )
        return opt
