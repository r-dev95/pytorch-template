"""This is the template module.
"""

import argparse
from logging import getLogger
from pathlib import Path
from typing import Any

from lib.common.decorator import process_time, save_params_log
from lib.common.file import load_yaml
from lib.common.log import SetLogging
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


class Tmp:
    """Tmp class

    sphinx docstring example:
    =========================

    link in file:
    -------------

    class     = :class:`Tmp`

    func      = :func:`main`

    method    = :meth:`tmp_method`

    attribute = :attr:`tmp_val`

    module    = :mod:`lib`

    code block:
    -----------

    .. code-block:: python

        def func():
            ...
            return rtn

    comments:
    ---------

    .. attention::

        description

    .. caution::

        description

    .. danger::

        description

    .. error::

        description

    .. hint::

        description

    .. important::

        description

    .. note::

        description

    .. tip::

        description

    .. warning::

        description

    .. admonition:: something

        description
    """
    tmp_val = None

    def __init__(self) -> None:
        pass

    def tmp_method(self) -> None:
        pass


@save_params_log(fname=f'log_params_{Path(__file__).stem}.yaml')
@process_time(print_func=LOGGER.info)
def main(params: dict[str, Any]) -> dict[str, Any]:
    """Main.

    This function is decorated by ``@save_params_log`` and ``@process_time``.

    Args:
        params (dict[str, Any]): parameters.

    Returns:
        dict[str, Any]: parameters.
    """
    return params


def set_params() -> dict[str, Any]:
    """Sets the command line arguments and file parameters.

    *   Set only common parameters as command line arguments.
    *   Other necessary parameters are set in the file parameters.
    *   Use a yaml file. (:func:`lib.common.file.load_yaml`)

    Returns:
        dict[str, Any]: parameters.

    .. attention::

        Command line arguments are overridden by file parameters.
        This means that if you want to set everything using file parameters,
        you don't necessarily need to use command line arguments.
    """
    # set the command line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        f'--{K.HANDLER}',
        default=[True, True], type=bool, nargs=2,
        help=(
            f'The log handler flag to use.\n'
            f'True: set handler, False: not set handler\n'
            f'ex) --{K.HANDLER} arg1 arg2 (arg1: stream handler, arg2: file handler)'
        ),
    )
    parser.add_argument(
        f'--{K.LEVEL}',
        default=[20, 20], type=int, nargs=2, choices=[10, 20, 30, 40, 50],
        help=(
            f'The log level.\n'
            f'DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50\n'
            f'ex) --{K.LEVEL} arg1 arg2 (arg1: stream handler, arg2: file handler)'
        ),
    )
    parser.add_argument(
        f'--{K.PARAM}',
        default='param/param.yaml', type=str,
        help=('The parameter file path.'),
    )
    parser.add_argument(
        f'--{K.RESULT}',
        default='result', type=str,
        help=('The directory path to save the results.'),
    )

    params = vars(parser.parse_args())

    # set the file parameters.
    if params.get(K.PARAM):
        fpath = Path(params[K.PARAM])
        if fpath.is_file():
            params.update(load_yaml(fpath=fpath))

    return params


if __name__ == '__main__':
    # set the parameters.
    params = set_params()
    # set the logging configuration.
    PARAM_LOG.HANDLER[PARAM_LOG.SH] = params[K.HANDLER][0]
    PARAM_LOG.HANDLER[PARAM_LOG.FH] = params[K.HANDLER][1]
    PARAM_LOG.LEVEL[PARAM_LOG.SH] = params[K.LEVEL][0]
    PARAM_LOG.LEVEL[PARAM_LOG.FH] = params[K.LEVEL][1]
    SetLogging(logger=LOGGER, param=PARAM_LOG)

    if params.get(K.RESULT):
        Path(params[K.RESULT]).mkdir(parents=True, exist_ok=True)

    main(params=params)
