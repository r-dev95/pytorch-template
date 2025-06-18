"""This is the module that tests decorator.py.
"""

import re
import shutil
from logging import getLogger
from pathlib import Path

import pytest
from _pytest.capture import MultiCapture

from lib.common import decorator
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestProcessTime:
    """Tests :func:`decorator.process_time`.
    """
    ptns = [
        r'^==================================================$',
        r'^\[START\]\[func\]$',
        r'^==================================================$',
        r'^==================================================$',
        r'^\[END\]\[func\] \d*.\d*e-\d*sec|\[END\]\[func\] \d*.\d*sec$',
        r'^==================================================$',
    ]

    def test(self, capfd: MultiCapture):
        """Tests that no errors are raised.

        *   The print output is correct.
        """
        @decorator.process_time(print_func=print)
        def func():
            return
        func()

        outs, errs = capfd.readouterr()
        assert not errs
        for out, ptn in zip(outs.split('\n'), self.ptns):
            assert re.match(pattern=ptn, string=out)


class TestSaveParamsLog:
    """Tests :func:`decorator.save_params_log`.
    """
    params = {
        K.RESULT: 'result',
    }
    suffixes = ['.yml', '.yaml', '.json', '.toml']

    @pytest.fixture(scope='class')
    def proc(self):
        yield
        shutil.rmtree(self.params[K.RESULT])

    def test(self, proc):
        """Tests that no errors are raised.

        *   A file is output.
        """
        for suffix in self.suffixes:
            fname = Path(f'log_param{suffix}')

            @decorator.save_params_log(fname=fname)
            def func():
                return self.params
            func()

            assert Path(self.params[K.RESULT], fname).is_file()
