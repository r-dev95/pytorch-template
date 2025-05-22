"""This is the module that load and write files.
"""

import json
from logging import getLogger
from pathlib import Path
from typing import Any

import tomlkit
import yaml

from lib.common.define import ParamLog
from lib.common.process import recursive_replace

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


# -----------------------------------------------
# json
# -----------------------------------------------
def dump_json(data: Any, fpath: Path, mode: str = 'w') -> None:  # noqa: ANN401
    """Writes json files.

    Args:
        data (Any): writing data.
        fpath (str): file path.
        mode (str): write mode.
    """
    with fpath.open(mode=mode) as f:
        json.dump(obj=data, fp=f)


def load_json(fpath: Path, mode: str = 'rb') -> dict[str, Any]:
    """Loads json files.

    Args:
        fpath (str): file path.
        mode (str): load mode.

    Returns:
        dict[str, Any]: loaded data.
    """
    with fpath.open(mode=mode) as f:
        data = json.load(fp=f)
    return data


# -----------------------------------------------
# yaml
# -----------------------------------------------
def dump_yaml(data: Any, fpath: Path, mode: str = 'w') -> None:  # noqa: ANN401
    """Writes yaml files.

    Args:
        data (Any): writing data.
        fpath (str): file path.
        mode (str): write mode.
    """
    with fpath.open(mode=mode) as f:
        yaml.dump(data=data, stream=f, sort_keys=False)


def load_yaml(fpath: Path, mode: str = 'rb') -> dict[str, Any]:
    """Loads yaml files.

    Args:
        fpath (str): file path.
        mode (str): load mode.

    Returns:
        dict[str, Any]: loaded data.
    """
    with fpath.open(mode=mode) as f:
        data = yaml.safe_load(stream=f)
    return data


# -----------------------------------------------
# toml
# -----------------------------------------------
def dump_toml(data: Any, fpath: Path, mode: str = 'w') -> None:  # noqa: ANN401
    """Writes toml files.

    Args:
        data (Any): writing data.
        fpath (str): file path.
        mode (str): write mode.
    """
    data = recursive_replace(data=data, fm_val=None, to_val='None')
    with fpath.open(mode=mode) as f:
        tomlkit.dump(data=data, fp=f, sort_keys=False)


def load_toml(fpath: Path, mode: str = 'rb') -> dict[str, Any]:
    """Loads toml files.

    Args:
        fpath (str): file path.
        mode (str): load mode.

    Returns:
        dict[str, Any]: loaded data.

    .. note::

        If you want to specify ``None`` , specify it as a string in toml file.
    """
    with fpath.open(mode=mode) as f:
        data = tomlkit.load(f)
    data = recursive_replace(data=data, fm_val='None', to_val=None)
    return data
