"""This is the module that makes TFRecord data.
"""  # noqa: INP001

import argparse
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import torch as to
import torchvision
import webdataset as wds

from lib.common.decorator import process_time
from lib.common.log import SetLogging
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


#: str: temporary path name
TMP_DPATH: str = 'tmp'

#: dict[str, Callable]: data loader
LOADER: dict[str, Callable] = {
    'mnist': torchvision.datasets.MNIST,
    'fashion_mnist': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
}


@dataclass
class ParamDataset:
    """Defines the parameters.
    """
    tmp_dpath: str
    dpath: Path
    loader: Callable
    maxcnt: int = 10000


def write_exmaple(args: ParamDataset) -> None:
    """Writes shard data (webdataset).

    Args:
        args (ParamDataset): parameters.
    """
    fpath = args.dpath
    loader = args.loader
    maxcnt = args.maxcnt

    with wds.ShardWriter(pattern=fpath.as_posix(), maxcount=maxcnt) as writer:
        for i, (x, y) in enumerate(loader):
            inputs = np.squeeze(x.numpy())
            labels = y.numpy()
            data = {
                '__key__': f'{i:06}',
                'inputs.npy': inputs,
                'labels.npy': labels,
            }
            writer.write(obj=data)


def worker(args: ParamDataset) -> None:
    """Makes shard data (webdataset).

    *   Make shard data (webdataset) loading data from the following function.

        *   ``torchvision.datasets.MNIST``
        *   ``torchvision.datasets.FashionMNIST``
        *   ``torchvision.datasets.CIFAR10``
        *   ``torchvision.datasets.CIFAR100``

    Args:
        args (ParamDataset): parameters.
    """
    tmp_dpath = args.tmp_dpath
    dpath = args.dpath
    loader = args.loader

    # training data
    args.dpath = Path(dpath, 'train', '%05d.tar')
    args.dpath.parent.mkdir(parents=True, exist_ok=True)
    args.loader = to.utils.data.DataLoader(
        loader(
            root=tmp_dpath,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
    )
    write_exmaple(args=args)

    # test data
    args.dpath = Path(dpath, 'test', '%05d.tar')
    args.dpath.parent.mkdir(parents=True, exist_ok=True)
    args.loader = to.utils.data.DataLoader(
        loader(
            root=tmp_dpath,
            train=False,
            download=False,
            transform=torchvision.transforms.ToTensor(),
        ),
    )
    write_exmaple(args=args)


@process_time(print_func=LOGGER.info)
def main(params: dict[str, Any]) -> None:
    """Main.

    Args:
        params (dict[str, Any]): parameters.
    """
    if 'all' in params[K.DATA]:
        params[K.DATA] = list(LOADER.keys())

    args = []
    tmp_dpath = Path(params[K.RESULT], TMP_DPATH)
    for data_kind in params[K.DATA]:
        dpath = Path(params[K.RESULT], data_kind)
        _args = ParamDataset(
            tmp_dpath=tmp_dpath,
            dpath=dpath,
            loader=LOADER[data_kind],
        )
        args.append(_args)

    with ProcessPoolExecutor(max_workers=params['max_workers']) as executer:
        executer.map(worker, args)

    shutil.rmtree(tmp_dpath)


def set_params() -> dict[str, Any]:
    """Sets the command line arguments.

    Returns:
        dict[str, Any]: parameters.
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
    parser.add_argument(
        f'--{K.DATA}',
        default='', type=str, nargs='+', choices=['all', LOADER.keys()],
        help=('The type of data to download.'),
    )
    parser.add_argument(
        f'--{K.MAX_WORKERS}', default=8, type=int,
        help=('The number of download workers.'),
    )

    params = vars(parser.parse_args())

    # # set the file parameters.
    # if params.get(K.PARAM):
    #     fpath = Path(params[K.PARAM])
    #     if fpath.is_file():
    #         params.update(load_yaml(fpath=fpath))

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
