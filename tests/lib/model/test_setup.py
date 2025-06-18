"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.model import setup

sys.path.append('../tests')
from define import Layer

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetupModel:
    """Tests :class:`setup.SetupModel`.
    """
    layers = ['flatten', 'linear_1', 'linear_2', 'linear_3', 'conv2d_1', 'conv2d_2', 'maxpool2d', 'relu']
    kinds = [
        # MLP
        ['flatten', 'linear_1', 'relu', 'linear_2'],
        # CNN
        ['conv2d_1', 'relu', 'conv2d_2', 'relu', 'maxpool2d', 'flatten', 'linear_3', 'relu', 'linear_2'],
    ]
    params = {
        K.MODEL: {K.KIND: 'simple'},
        K.LAYER: {
            K.KIND: kinds[0],
            layers[0]: Layer.FLATTEN,
            layers[1]: Layer.LINEAR_1,
            layers[2]: Layer.LINEAR_2,
            layers[3]: Layer.LINEAR_3,
            layers[4]: Layer.CONV2D_1,
            layers[5]: Layer.CONV2D_2,
            layers[6]: Layer.MAXPOOL2D,
            layers[7]: Layer.RELU,
        },
        K.DEVICE: 'cpu',
    }
    params_raise = {
        K.MODEL: {
            K.KIND: '',
            '': {},
        },
    }

    labels = [
        "<class 'lib.model.simple.SimpleModel'>",
        "<class 'lib.model.simple.SimpleModel'>",
    ]
    all_log = [
        ('main', ERROR, f'SetupModel class does not have a method "{params_raise[K.MODEL][K.KIND]}" that sets the model.'),
        ('main', ERROR, f'The available model are:'),
    ]
    for key in setup.SetupModel(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        for kind, label in zip(self.kinds, self.labels):
            self.params[K.LAYER][K.KIND] = kind
            _class = setup.SetupModel(params=self.params).setup()
            print(f'{type(_class)=}')
            assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        with pytest.raises(ValueError):
            setup.SetupModel(params=self.params_raise).setup()

        assert caplog.record_tuples == self.all_log
