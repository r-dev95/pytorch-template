"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_pytorch/')
from template_pytorch.lib.common.define import ParamKey, ParamLog
from template_pytorch.lib.metrics import setup

sys.path.append('../tests')
from define import Metrics

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetupMetrics:
    """Tests :class:`setup.SetupMetrics`.
    """
    kinds = ['mse', 'bacc', 'macc']
    params = {
        K.METRICS: {
            K.KIND: kinds,
            kinds[0]: Metrics.MSE,
            kinds[1]: Metrics.BACC,
            kinds[2]: Metrics.MACC,
        },
    }
    params_raise = {
        K.METRICS: {
            K.KIND: [''],
            '': {},
        },
    }

    labels = "<class 'torchmetrics.collections.MetricCollection'>"
    all_log = [
        ('main', ERROR, f'SetupMetrics class does not have a method "{params_raise[K.METRICS][K.KIND][0]}" that sets the metrics.'),
        ('main', ERROR, f'The available metrics are:'),
    ]
    for key in setup.SetupMetrics(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        classes = setup.SetupMetrics(params=self.params).setup()
        print(f'{type(classes)=}')
        assert str(type(classes)) == self.labels

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        with pytest.raises(ValueError):
            setup.SetupMetrics(params=self.params_raise).setup()

        assert caplog.record_tuples == self.all_log
