"""This is the module that sets up callbacks.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any

from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from lib.common.define import ParamFileName, ParamKey, ParamLog

K = ParamKey()
PARAM_FILE_NAME = ParamFileName()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any], func: dict[str, Callable]) -> None:
    """Checks the :class:`SetupCallbacks` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    for kind in params[K.CB][K.KIND]:
        if kind not in func:
            error = True
            LOGGER.error(
                f'SetupCallbacks class does not have a method "{kind}" that '
                f'sets the callbacks.',
            )
    if error:
        LOGGER.error('The available callbacks are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupCallbacks:
    """Sets up callbacks.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'ms': self.ms,
            'mcp': self.mcp,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> Callable:
        """Sets up callbacks.

        Returns:
            Callable: callbacks class.
        """
        callbacks = []
        for kind in self.params[K.CB][K.KIND]:
            self._params = self.params[K.CB][kind]
            callbacks.append(self.func[kind]())
        return callbacks

    def ms(self) -> Callable:
        """Sets ``lightning.pytorch.callbacks.ModelSummary``.

        Returns:
            Callable: callbacks class.
        """
        callbacks = ModelSummary(
            max_depth=self._params['max_depth'],
        )
        return callbacks

    def mcp(self) -> Callable:
        """Sets ``lightning.pytorch.callbacks.ModelCheckpoint``.

        Returns:
            Callable: callbacks class.
        """
        callbacks = ModelCheckpoint(
            dirpath=self.params[K.RESULT],
            filename=PARAM_FILE_NAME.WIGHT,
            monitor=self._params['monitor'],
            verbose=self._params['verbose'],
            save_last=self._params['save_last'],
            save_top_k=self._params['save_top_k'],
            save_weights_only=self._params['save_weights_only'],
            mode=self._params['mode'],
            auto_insert_metric_name=self._params['auto_insert_metric_name'],
            every_n_train_steps=self._params['every_n_train_steps'],
            train_time_interval=self._params['train_time_interval'],
            every_n_epochs=self._params['every_n_epochs'],
            save_on_train_epoch_end=self._params['save_on_train_epoch_end'],
            enable_version_counter=self._params['enable_version_counter'],
        )
        return callbacks
