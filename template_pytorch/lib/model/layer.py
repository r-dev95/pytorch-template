"""This is the module that sets up model layers.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any

import torch as to

from lib.common.define import ParamKey, ParamLog

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any], func: dict[str, Callable]) -> None:
    """Checks the :class:`SetupLayer` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    for kind in params[K.LAYER][K.KIND]:
        layer = kind.split('_')[0]
        if layer not in func:
            error = True
            LOGGER.error(
                f'SetupLayer class does not have a method "{kind}" that '
                f'sets the model layer.',
            )
    if error:
        LOGGER.error('The available model layer are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupLayer:
    """Sets up the model layer.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'flatten': self.flatten,
            'linear': self.linear,
            'conv2d': self.conv2d,
            'maxpool2d': self.maxpool2d,
            'relu': self.relu,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> list[Callable]:
        """Sets up model layers.

        Returns:
            list[Callable]: list of model layers.
        """
        layers = []
        for layer in self.params[K.LAYER][K.KIND]:
            _layer = layer.split('_')[0]
            self._params = self.params[K.LAYER][layer]
            layers.append(
                self.func[_layer](),
            )
        return layers

    def flatten(self) -> Callable:
        """Sets ``to.nn.Flatten``.

        Returns:
            Callable: model layer class.
        """
        layer = to.nn.Flatten(
            start_dim=self._params['start_dim'],
            end_dim=self._params['end_dim'],
        )
        return layer

    def linear(self) -> Callable:
        """Sets ``to.nn.Linear``.

        Returns:
            Callable: model layer class.
        """
        layer = to.nn.Linear(
            in_features=self._params['in_features'],
            out_features=self._params['out_features'],
            bias=self._params['bias'],
            # device=self._params['device'],
            # dtype=self._params['dtype'],
        )
        return layer

    def conv2d(self) -> Callable:
        """Sets ``to.nn.Conv2d``.

        Returns:
            Callable: model layer class.
        """
        layer = to.nn.Conv2d(
            in_channels=self._params['in_channels'],
            out_channels=self._params['out_channels'],
            kernel_size=self._params['kernel_size'],
            stride=self._params['stride'],
            padding=self._params['padding'],
            dilation=self._params['dilation'],
            groups=self._params['groups'],
            bias=self._params['bias'],
            padding_mode=self._params['padding_mode'],
            # device=self._params['device'],
            # dtype=self._params['dtype'],
        )
        return layer

    def maxpool2d(self) -> Callable:
        """Sets ``to.nn.MaxPool2d``.

        Returns:
            Callable: model layer class.
        """
        layer = to.nn.MaxPool2d(
            kernel_size=self._params['kernel_size'],
            stride=self._params['stride'],
            padding=self._params['padding'],
            dilation=self._params['dilation'],
            return_indices=self._params['return_indices'],
            ceil_mode=self._params['ceil_mode'],
        )
        return layer

    def relu(self) -> Callable:
        """Sets ``to.nn.ReLU``.

        Returns:
            Callable: model layer class.
        """
        layer = to.nn.ReLU(
            inplace=self._params['inplace'],
        )
        return layer
