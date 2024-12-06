"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_pytorch/')
from template_pytorch.lib.callbacks import setup
from template_pytorch.lib.common.define import ParamKey, ParamLog

sys.path.append('../tests/')
from define import CB

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetupCallbacks:
    """Tests :class:`setup.SetupCallbacks`.
    """
    kinds = ['ms', 'mcp']
    params = {
        K.RESULT: '.',
        K.CB: {
            K.KIND: kinds,
            kinds[0]: CB.MS,
            kinds[1]: CB.MCP,
        },
    }
    params_raise = {
        K.CB: {
            K.KIND: [''],
            '': {},
        },
    }

    labels = [
        "<class 'lightning.pytorch.callbacks.model_summary.ModelSummary'>",
        "<class 'lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint'>",
    ]
    all_log = [
        ('main', ERROR, f'SetupCallbacks class does not have a method "{params_raise[K.CB][K.KIND][0]}" that sets the callbacks.'),
        ('main', ERROR, f'The available callbacks are:'),
    ]
    for key in setup.SetupCallbacks(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        classes = setup.SetupCallbacks(params=self.params).setup()
        for _class, label in zip(classes, self.labels):
            print(f'{type(_class)=}')
            assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        with pytest.raises(ValueError):
            setup.SetupCallbacks(params=self.params_raise).setup()

        assert caplog.record_tuples == self.all_log
