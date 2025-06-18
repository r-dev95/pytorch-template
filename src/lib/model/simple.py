"""This is the module that builds simple model.
"""

from logging import getLogger
from typing import Any, override

import torch as to

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.model.layer import SetupLayer

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any]) -> None:
    """Checks the :class:`SimpleModel` parameters.

    Args:
        params (dict[str, Any]): parameters.
    """
    error = False # error: True
    keys = [K.DEVICE]
    for key in keys:
        if key not in params:
            error = True
            LOGGER.error(f'The key "{key}" for variable "params" is missing.')
    if error:
        raise ValueError


class SimpleModel(to.nn.Module):
    """Builds the following simple model.

    *   MLP (Multi Layer Perceptron)
    *   CNN (Convolutional Neural Network)

    Args:
        params (dict[str, Any]): parameters.
    """
    def __init__(self, params: dict[str, Any]) -> None:
        check_params(params=params)
        super().__init__()

        model_layers = SetupLayer(params=params).setup()
        self.model_layers = to.nn.Sequential()
        for module in model_layers:
            self.model_layers.append(module=module)

        self = self.to(device=params[K.DEVICE])  # noqa: PLW0642
        self = to.jit.script(self)  # noqa: PLW0642

    @override
    def forward(self, x: to.Tensor) -> to.Tensor:
        """Outputs the model predictions.

        This function is decorated by ``@override``.

        Args:
            x (to.Tensor): input.

        Returns:
            to.Tensor: output.
        """
        x = self.model_layers(x)
        return x
