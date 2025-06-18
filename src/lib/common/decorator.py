"""This is the module that defines the decorator.
"""

import datetime
import functools
import time
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any

from lib.common.file import dump_json, dump_toml, dump_yaml
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog, ZoneInfo

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def process_time(print_func: Callable = print) -> Callable:
    """Prints the processing time.

    Args:
        print_func (Callable): standard output function.

            ``print`` or logging (``.debug``, ``.info``, ... ) etc...

    Returns:
        Callable: ``_process_time`` function in this function.

    .. code-block:: python

        @process_time(print_func=print)
        def func():
            ...
            return rtn

        # ==================================================
        # [START][main]
        # ==================================================
        # ...
        # ==================================================
        # [END][main] 12.345678sec
        # ==================================================
    """
    def _process_time(func: Callable) -> Callable:
        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            print_func(f'{'=' * 50}')
            print_func(f'[START][{func.__name__}]')
            print_func(f'{'=' * 50}')
            rtn = func(*args, **kwargs)
            end_time = time.perf_counter()
            print_func(f'{'=' * 50}')
            print_func(f'[END][{func.__name__}] {end_time - start_time:>.6}sec')
            print_func(f'{'=' * 50}')
            return rtn
        return _wrapper
    return _process_time


def save_params_log(fname: str = 'log_params.yaml') -> Callable:
    """Saves parameters.

    *   The execution start date and time, end date and time, and processing time are
        also saved.
    *   The return value of the function to which the decorator is applied is assumed to
        be a dictionary.
        If it is not a dictionary type, it will not be saved.
    *   If the dictionary key has "result" (data save directory path), it will be
        saved in that directory.
        If not, it saves it in the current directory.

    Args:
        fname (str): file
            (The extension is ``.yml``, ``.yaml``, ``.json``, ``.toml``.)

    Returns:
        Callable: ``_save_params_log`` function in this function.

    .. code-block:: python

        @save_params_log(fname='log.yaml')
        def func(params):
            ...
            return params

        params = {
            'aaa': 'abc',
            'bbb': 1,
            'ccc': [1],
            'ddd': [1, 2],
            'result': 'path/to/dir',
        }
        func(params=params)

    Output (path/to/dir/log.yaml):

    .. code-block:: yaml

        aaa: abc
        bbb: 1
        ccc:
        - 1
        ddd:
        - 1
        - 2
        result: path/to/dir
        start_datetime: 2024-1-1 00:00:00.000000
        end_datetime: 2024-1-1 00:00:12.345678
        process_time: 12.34567891234567
    """
    def _save_params_log(func: Callable) -> Callable:
        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            start_time = time.perf_counter()
            start_datetime = datetime.datetime.now(tz=ZoneInfo)
            rtn = func(*args, **kwargs)
            end_time = time.perf_counter()
            end_datetime = datetime.datetime.now(tz=ZoneInfo)
            if not isinstance(rtn, dict):
                return rtn
            rtn['start_datetime'] = start_datetime
            rtn['end_datetime'] = end_datetime
            rtn['process_time'] = end_time - start_time
            fpath = Path(fname)
            if rtn.get(K.RESULT):
                Path(rtn[K.RESULT]).mkdir(parents=True, exist_ok=True)
                fpath = Path(rtn[K.RESULT], fpath)
            if fpath.suffix in ['.yml', '.yaml']:
                dump_yaml(data=rtn, fpath=fpath)
            elif fpath.suffix in ['.json']:
                rtn['start_datetime'] = str(start_datetime)
                rtn['end_datetime'] = str(end_datetime)
                dump_json(data=rtn, fpath=fpath)
            elif fpath.suffix in ['.toml']:
                dump_toml(data=rtn, fpath=fpath)
            return rtn
        return _wrapper
    return _save_params_log
