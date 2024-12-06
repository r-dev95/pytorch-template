"""This is the module that defines the common process.
"""

import collections
import random
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as to

from lib.common.define import ParamFileName, ParamKey, ParamLog

K = ParamKey()
PARAM_FILE_NAME = ParamFileName()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def get_device_type() -> str:
    """Gets the device type (cpu or mpu or cuda).

    Returns:
        str: device type.
    """
    device = (
        'cuda'
        if to.cuda.is_available()
        else 'mps'
        if to.backends.mps.is_available()
        else 'cpu'
    )
    return device


def fix_random_seed(seed: int) -> None:
    """Fixes the random seed to ensure reproducibility of experiment.

    Args:
        seed (int): random seed.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    to.manual_seed(seed)
    to.backends.cudnn.benchmark = False
    to.backends.cudnn.deterministic = True


def set_weight(params: dict[str, Any], model: to.nn.Module) -> to.nn.Module:
    """Sets the model weight.

    Args:
        params (dict[str, Any]): parameters.
        model (to.nn.Module): model class.

    Returns:
        to.nn.Module: weighted model class.
    """
    nums = []
    for ver in Path(params[K.RESULT]).glob('version_*'):
        if ver.is_dir():
            try:
                nums.append(int(ver.stem.split(sep='_')[1]))
            except ValueError:
                LOGGER.warning(f'{ver=} is skipped.')
    if len(nums) != 0:
        ver = f'version_{max(nums)}'

    fpath = Path(params[K.RESULT], ver, PARAM_FILE_NAME.LOSS)
    df = pd.read_csv(fpath)
    idx_min = int(df['val_loss'].argmin())
    idx_min = int(df.loc[idx_min]['epoch'])

    fpath = list(Path(params[K.RESULT]).glob('*.ckpt'))[idx_min]
    ckpt = to.load(f=fpath, weights_only=True)['state_dict']
    new_ckpt = collections.OrderedDict()
    for key in ckpt:
        if key.startswith('model.'):
            new_key = key.replace('model.', '')
        new_ckpt[new_key] = ckpt[key]
    model.load_state_dict(new_ckpt)
    return model


def recursive_replace(data: Any, fm_val: Any, to_val: Any) -> Any:  # noqa: ANN401
    """Performs a recursive replacement.

    Args:
        data (Any): data before replacement.
        fm_val (Any): value before replacement.
        to_val (Any): value after replacement.

    Returns:
        Any: data after replacement.
    """
    if isinstance(data, dict):
        return {
            key: recursive_replace(
                data=val,
                fm_val=fm_val,
                to_val=to_val,
            ) for key, val in data.items()
        }
    if isinstance(data, list):
        return [
            recursive_replace(
                data=val,
                fm_val=fm_val,
                to_val=to_val,
            ) for val in data
        ]
    if isinstance(data, tuple):
        return tuple(
            recursive_replace(
                data=val,
                fm_val=fm_val,
                to_val=to_val,
            ) for val in data
        )
    if isinstance(data, set):
        return {
            recursive_replace(
                data=val,
                fm_val=fm_val,
                to_val=to_val,
            ) for val in data
        }
    if data == fm_val:
        return to_val
    return data


def parse_tar_fname_number(dpath: Path) -> str:
    """Parse the highest and lowest numbers from a tar file whose filenames are numeric.

    Args:
        dpath (Path): the tar file parent path.

    Returns:
        str: parsed load tar file pattern.
    """
    files = list(dpath.glob('*.tar'))
    nums = []
    for file in files:
        print(f'{file=}')
        try:
            nums.append(int(file.stem))
        except ValueError:
            LOGGER.warning(
                f'"{file}" A will not be loaded '
                'because it contains non-numeric characters.',
            )
    max_idx = np.argmax(nums)
    min_idx = np.argmin(nums)
    file_pattern = f'{{{files[min_idx].stem}..{files[max_idx].stem}}}' + '.tar'
    return file_pattern
