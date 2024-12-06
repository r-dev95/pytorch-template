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
from lib.common.define import ParamKey, ParamLog
from lib.common.log import SetLogging

K = ParamKey()
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
    """main.

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
    parser = argparse.ArgumentParser()
    # log level (idx=0: stream handler, idx=1: file handler)
    # (DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50)
    choices = [10, 20, 30, 40, 50]
    parser.add_argument('--level', default=[20, 20], type=int, nargs=2, choices=choices)
    # directory path (data save)
    parser.add_argument('--result', default='', type=str)
    # data
    choices = ['all']
    choices.extend(list(LOADER.keys()))
    parser.add_argument('--data', default='', type=str, nargs='+', choices=choices)
    # max workers
    parser.add_argument('--max_workers', default=8, type=int)

    params = vars(parser.parse_args())

    return params


if __name__ == '__main__':
    # set the parameters.
    params = set_params()
    # set the logging configuration.
    PARAM_LOG.LEVEL[PARAM_LOG.SH] = params[K.LEVEL][0]
    PARAM_LOG.LEVEL[PARAM_LOG.FH] = params[K.LEVEL][1]
    SetLogging(logger=LOGGER, param=PARAM_LOG)

    if K.RESULT in params:
        Path(params[K.RESULT]).mkdir(parents=True, exist_ok=True)

    main(params=params)
