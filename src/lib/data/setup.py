"""This is the module that sets up data.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any

from lib.common.define import ParamKey, ParamLog
from lib.data.base import BaseLoadData
from lib.data.cifar import Cifar
from lib.data.mnist import Mnist

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any], func: dict[str, Callable]) -> None:
    """Checks the :class:`SetupData` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    kind = params[K.DATA][K.KIND]
    if kind not in func:
        error = True
        LOGGER.error(
            f'SetupData class does not have a method "{kind}" that sets the data.',
        )
    if error:
        LOGGER.error('The available data are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupData:
    """Sets up data.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'mnist': self.mnist,
            'fashion_mnist': self.mnist,
            'cifar10': self.cifar,
            'cifar100': self.cifar,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> BaseLoadData:
        """Sets up data.

        Returns:
            BaseLoadData: data class.
        """
        kind = self.params[K.DATA][K.KIND]
        data = self.func[kind]()
        return data

    def mnist(self) -> Mnist:
        """Sets :class:`lib.data.mnist.Mnist` (mnist or fashion mnist).

        Returns:
            Mnist: data class.
        """
        data = Mnist(params=self.params)
        return data

    def cifar(self) -> Cifar:
        """Sets :class:`lib.data.cifar.Cifar` (cifar10 or cifar100).

        Returns:
            Cifar: data class.
        """
        data = Cifar(params=self.params)
        return data
