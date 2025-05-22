"""This is the module that sets up model.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any

from lib.common.define import ParamFileName, ParamKey, ParamLog
from lib.model.simple import SimpleModel

K = ParamKey()
PARAM_FILE_NAME = ParamFileName()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any], func: dict[str, Callable]) -> None:
    """Checks the :class:`SetupModel` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    kind = params[K.MODEL][K.KIND]
    if kind not in func:
        error = True
        LOGGER.error(
            f'SetupModel class does not have a method "{kind}" that sets the model.',
        )
    if error:
        LOGGER.error('The available model are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupModel:
    """Sets up model.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'simple': self.simple,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> Callable:
        """Sets up model.

        Returns:
            Callable: model class.
        """
        kind = self.params[K.MODEL][K.KIND]
        model = self.func[kind]()
        return model

    def simple(self) -> Callable:
        """Sets :class:`lib.model.simple.SimpleModel`.

        Returns:
            Callable: model class.
        """
        model = SimpleModel(params=self.params)
        return model
