"""This is the module that tests eval.py.
"""

import shutil
import sys
from logging import getLogger
from pathlib import Path

import pytest
import torch as to
from pytest_mock import MockerFixture

import dataset
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.common.process import parse_tar_fname_number
from lib.data import base

sys.path.append('../tests')
from define import DATA_PARENT_DPATH

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestWriteExmaple:
    """Tests :func:`base.write_exmaple`.
    """
    params = {
        K.PROCESS: {
            K.KIND: [],
        },
        K.BATCH: 1,
        K.SHUFFLE: None,
        K.FILE_PATTERN: 'result/{00000..00001}.tar',
        K.DEVICE: 'cpu',
        K.NUM_WORKERS: 4,
    }

    inputs = [to.Tensor([1., 2.]), to.Tensor([3., 4.])]
    labels = [to.Tensor([0.]), to.Tensor([1.])]

    @pytest.fixture(scope='class')
    def proc(self):
        def loader():
            for x, y in zip(self.inputs, self.labels):  # noqa: UP028
                yield x, y

        base.BaseLoadData.n_data = 2

        fpath = Path(self.params[K.FILE_PATTERN])
        fpath.parent.mkdir(parents=True, exist_ok=True)
        args = dataset.ParamDataset(
            tmp_dpath=None,
            dpath=Path(fpath.parent, '%05d.tar'),
            loader=loader(),
            maxcnt=1,
        )
        yield args
        shutil.rmtree(fpath.parent)

    def test(self, proc, mocker: MockerFixture):
        """Tests that no errors are raised.

        *   Data can be written to the file and load correctly.
        *   The :class:`base.BaseLoadData` is also being tested simultaneously in
            loading tests.
        """
        dataset.write_exmaple(args=proc)

        mocker.patch.object(base.BaseLoadData, 'set_model_il_shape', return_value=None)
        loader = base.BaseLoadData(params=self.params).make_loader_example()
        for i, (x, y) in enumerate(loader):
            assert (self.inputs[i] == x.numpy()).all()
            assert (self.labels[i] == y.numpy()).all()


class TestDataset:
    """Tests :mod:`dataset`.
    """
    params = {
        K.RESULT: DATA_PARENT_DPATH,
        K.DATA: ['all'],
        'max_workers': 8,
    }

    def _parse_num_files(self, dpath: Path):
        num = (
            int(
                parse_tar_fname_number(dpath=dpath)
                .replace('.tar', '')
                .split('..')[1]
                .replace('}', ''),
            ) + 1
        )
        return num

    def test(self):
        """Tests that no errors are raised.
        """
        dataset.main(params=self.params)

        kinds = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
        for kind in kinds:
            # training data
            dpath = Path(self.params[K.RESULT], kind, 'train')
            num = self._parse_num_files(dpath=dpath)
            for i_num in range(num):
                fpath = Path(dpath, f'{i_num:05}.tar')
                assert fpath.is_file()
                assert fpath.stat().st_size > 0

            # test data
            dpath = Path(self.params[K.RESULT], kind, 'test')
            num = self._parse_num_files(dpath=dpath)
            for i_num in range(num):
                fpath = Path(dpath, f'{i_num:05}.tar')
                assert fpath.is_file()
                assert fpath.stat().st_size > 0
