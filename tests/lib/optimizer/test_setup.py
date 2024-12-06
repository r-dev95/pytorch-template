"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_pytorch/')
from template_pytorch.lib.common.define import ParamKey, ParamLog
from template_pytorch.lib.model.setup import SetupModel
from template_pytorch.lib.optimizer import setup

sys.path.append('../tests')
from define import Opt, Layer

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetupOpt:
    """Tests :class:`setup.SetupOpt`.
    """
    kinds = ['sgd', 'adam']
    params = {
        K.OPT: {
            K.KIND: kinds[0],
            kinds[0]: Opt.SGD,
            kinds[1]: Opt.ADAM,
        },
    }
    params_model = {
        K.MODEL: {K.KIND: 'simple'},
        K.LAYER: {
            K.KIND: ['flatten', 'linear_1', 'relu', 'linear_2'],
            'flatten': Layer.FLATTEN,
            'linear_1': Layer.LINEAR_1,
            'linear_2': Layer.LINEAR_2,
            'relu': Layer.RELU,
        },
        K.DEVICE: 'cpu',
    }
    params_raise = {
        K.OPT: {
            K.KIND: '',
            '': {},
        },
    }

    labels = [
        "<class 'torch.optim.sgd.SGD'>",
        "<class 'torch.optim.adam.Adam'>",
    ]
    all_log = [
        ('main', ERROR, f'SetupOpt class does not have a method "{params_raise[K.OPT][K.KIND]}" that sets the optimizer method.'),
        ('main', ERROR, f'The available optimizer method are:'),
    ]
    for key in setup.SetupOpt(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        for kind, label in zip(self.kinds, self.labels):
            model = SetupModel(params=self.params_model).setup()
            self.params[K.MODEL_PARAMS] = model.parameters()

            self.params[K.OPT][K.KIND] = kind
            _class = setup.SetupOpt(params=self.params).setup()
            print(f'{type(_class)=}')
            assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        with pytest.raises(ValueError):
            setup.SetupOpt(params=self.params_raise).setup()

        assert caplog.record_tuples == self.all_log
