"""This is the module that tests processor.py.
"""

import sys
from logging import ERROR, getLogger

import numpy as np
import pytest
import torch as to
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_pytorch/')
from template_pytorch.lib.common.define import ParamKey, ParamLog
from template_pytorch.lib.data import processor

sys.path.append('../tests')
from define import Proc

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestProcessor:
    """Tests :class:`processor.Processor`.
    """
    kinds = ['one_hot', 'rescale']
    params = {
        K.PROCESS: {
            K.KIND: kinds,
            kinds[0]: Proc.ONE_HOT,
            kinds[1]: Proc.RESCALE,
        },
    }
    params_raise = {
        K.PROCESS: {
            K.KIND: [''],
            '': {},
        },
    }

    all_log = [
        ('main', ERROR, f'Processor class does not have a method "{params_raise[K.PROCESS][K.KIND][0]}" that sets the processing method.'),
        ('main', ERROR, f'The available processing method are:'),
    ]
    for key in processor.Processor(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

    def test_one_hot(self):
        """Tests that no errors are raised.

        *   This class tests :class:`processor.Processor.one_hot`.
        *   The process result is correct.
        """
        self.params[K.PROCESS][K.KIND] = ['one_hot']
        _class = processor.Processor(params=self.params)

        x_ini = to.Tensor([[1, 2], [3, 4]])
        y_ini = to.Tensor([4, 9])
        x, y = _class.run(x=x_ini, y=y_ini)

        assert (x_ini == x).all()
        assert (np.identity(Proc.ONE_HOT['num_classes'])[[4, 9]] == y.numpy()).all()

    def test_rescale(self):
        """Tests that no errors are raised.

        *   This class tests :class:`processor.Processor.rescale`.
        *   The process result is correct.
        """
        self.params[K.PROCESS][K.KIND] = ['rescale']
        _class = processor.Processor(params=self.params)

        x_ini = to.Tensor([[1, 2], [3, 4]])
        y_ini = to.Tensor([4, 9])
        x, y = _class.run(x=x_ini, y=y_ini)

        assert (x_ini * Proc.RESCALE['scale'] == x.numpy()).all()
        assert (y_ini == y).all()

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        with pytest.raises(ValueError):
            processor.Processor(params=self.params_raise)

        assert caplog.record_tuples == self.all_log
