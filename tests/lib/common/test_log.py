"""This is the module that tests log.py.
"""

import shutil
from logging import getLogger
from pathlib import Path

import pytest

from lib.common import log
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

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

        # Addressed an issue in Windows where a log file was grabbed by a process and
        # could not be deleted.
        for handler in LOGGER.handlers[:]:
            LOGGER.removeHandler(hdlr=handler)
            handler.close()

        assert Path(PARAM_LOG.FPATH).is_file()
