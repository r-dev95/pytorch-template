"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger
from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_pytorch/')
from template_pytorch.lib.common.define import ParamKey, ParamLog
from template_pytorch.lib.data import setup

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetupData:
    """Tests :class:`setup.SetupData`.
    """
    kinds = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
    params = {
        K.DATA: {K.KIND: kinds[0]},
        K.DPATH: None,
        K.PROCESS: {K.KIND: []},
        K.BATCH: 10,
        K.SHUFFLE: None,
        K.DEVICE: 'cpu',
        K.NUM_WORKERS: 4,
    }
    params_raise = {
        K.DATA: {
            K.KIND: '',
        },
    }

    labels = [
        "<class 'lib.data.mnist.Mnist'>",
        "<class 'lib.data.mnist.Mnist'>",
        "<class 'lib.data.cifar.Cifar'>",
        "<class 'lib.data.cifar.Cifar'>",
    ]
    all_log = [
        ('main', ERROR, f'SetupData class does not have a method "{params_raise[K.DATA][K.KIND]}" that sets the data.'),
        ('main', ERROR, f'The available data are:'),
    ]
    for key in setup.SetupData(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        for kind, label in zip(self.kinds, self.labels):
            self.params[K.DATA][K.KIND] = kind
            self.params[K.DPATH] = Path('data', self.params[K.DATA][K.KIND], 'train')
            _class = setup.SetupData(params=self.params).setup()
            print(f'{type(_class)=}')
            assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        with pytest.raises(ValueError):
            setup.SetupData(params=self.params_raise).setup()

        assert caplog.record_tuples == self.all_log
