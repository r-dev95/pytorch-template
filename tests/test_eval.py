"""This is the module that tests eval.py.
"""

import sys
from logging import ERROR, INFO, WARNING, getLogger
from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture

import eval  # noqa: A004
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

sys.path.append('../tests')
from define import DATA_RESULT_DPATH, DATA_PARENT_DPATH, Layer, Loss, Metrics, Proc

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCheckParams:
    """Tests :func:`eval.check_params`.
    """
    params = {
        K.SEED: 0,
        K.RESULT: '.',
        K.EVAL: '.',
        K.BATCH: 1000,
        K.NUM_WORKERS: 4,
        K.DATA: '',
        K.PROCESS: '',
        K.MODEL: '',
        K.LAYER: '',
        K.LOSS: '',
        K.METRICS: '',
        K.CB: '',
    }
    params_raise = {
        K.SEED: None,
        K.RESULT: 'dummy',
        K.EVAL: 'dummy',
        K.BATCH: 0,
        K.NUM_WORKERS: 0,
    }

    all_log = [
        ('main', WARNING, f'params["{K.SEED}"] must be integer.'),
        ('main', WARNING, f'The random number seed is not fixed.'),
        ('main', ERROR  , f'params["{K.RESULT}"] is None or the directory does not exists.'),
        ('main', ERROR  , f'params["{K.EVAL}"] is None or the directory does not exists.'),
        ('main', ERROR  , f'params["{K.BATCH}"] must be greater than zero.'),
        ('main', ERROR  , f'params["{K.NUM_WORKERS}"] must be greater than zero.'),
    ]
    for key in [K.DATA, K.PROCESS, K.MODEL, K.LAYER, K.LOSS, K.METRICS]:
        all_log.append(('main', ERROR  , f'The key "{key}" for variable "params" is missing.'))

    def test(self):
        """Tests that no errors are raised.
        """
        eval.check_params(params=self.params)

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        caplog.set_level(INFO)
        with pytest.raises(ValueError):
            eval.check_params(params=self.params_raise)

        assert caplog.record_tuples == self.all_log


class TestEval:
    """Tests :class:`eval.evaluator`.
    """
    params = {
        K.SEED: 0,
        K.RESULT: DATA_RESULT_DPATH,
        K.EVAL: f'{DATA_PARENT_DPATH}/mnist/test',
        K.BATCH: 1000,
        K.NUM_WORKERS: 4,
        K.DATA: {K.KIND: 'mnist'},
        K.PROCESS: {
            K.KIND: ['one_hot', 'rescale'],
            'one_hot': Proc.ONE_HOT,
            'rescale': Proc.RESCALE,
        },
        K.MODEL: {K.KIND: 'simple'},
        K.LAYER: {
            K.KIND: ['flatten', 'linear_1', 'relu', 'linear_2'],
            'flatten': Layer.FLATTEN,
            'linear_1': Layer.LINEAR_1,
            'linear_2': Layer.LINEAR_2,
            'relu': Layer.RELU,
        },
        K.LOSS: {
            K.KIND: 'ce',
            'ce': Loss.CE,
        },
        K.METRICS: {
            K.KIND: ['mse'],
            'mse': Metrics.MSE,
        },
    }

    @pytest.fixture(scope='class')
    def proc(self):
        yield
        Path(self.params[K.RESULT], 'log_params_eval.yaml').unlink()

    def test(self, proc):
        """Tests that no errors are raised.
        """
        eval.main(params=self.params)
