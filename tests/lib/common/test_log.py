"""This is the module that tests log.py.
"""

import shutil
import sys
from logging import getLogger
from pathlib import Path

import pytest

sys.path.append('../template_pytorch/')
from template_pytorch.lib.common import log
from template_pytorch.lib.common.define import ParamKey, ParamLog

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetLogging:
    """Tests :class:`log.SetLogging`.
    """

    @pytest.fixture(scope='class')
    def proc(self):
        dpath = Path('log')
        dpath.mkdir(parents=True, exist_ok=True)
        yield
        shutil.rmtree(dpath)

    def test(self, proc):
        """Tests that no errors are raised.

        *   A log file is output.
        """
        log.SetLogging(logger=LOGGER, param=PARAM_LOG)

        assert Path(PARAM_LOG.FPATH).is_file()
