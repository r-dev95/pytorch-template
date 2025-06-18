"""This is the module that tests base.py.
"""

from logging import ERROR, INFO, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.model import base

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCheckParams:
    """Tests :func:`base.check_params`.
    """
    classes = {
        K.MODEL: '',
        K.OPT: '',
        K.LOSS: '',
        K.METRICS: '',
    }
    classes_raise = {}

    all_log = []
    for key in [K.MODEL, K.OPT, K.LOSS, K.METRICS]:
        all_log.append(('main', ERROR  , f'The key "{key}" for variable "classes" is missing.'))

    def test(self):
        """Tests that no errors are raised.
        """
        base.check_params(classes=self.classes)

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        caplog.set_level(INFO)
        with pytest.raises(ValueError):
            base.check_params(classes=self.classes_raise)

        assert caplog.record_tuples == self.all_log


class TestBaseModel:
    """Tests :class:`base.BaseModel`.

    *   This class is a base class and is not called directly, so it will not be tested.
    *   It is tested by :class:`test_train.TestTrain`
        and :class:`test_eval.TestEval` class.
    """
