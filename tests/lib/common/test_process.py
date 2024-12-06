"""This is the module that tests process.py.
"""

import random
import sys
from logging import getLogger
from pathlib import Path

import numpy as np
import torch as to

sys.path.append('../template_pytorch/')
from template_pytorch.lib.common import process
from template_pytorch.lib.common.define import ParamKey, ParamLog
from template_pytorch.lib.model.setup import SetupModel

sys.path.append('../tests')
from define import DATA_RESULT_DPATH, Layer

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestGetDeviceType:
    """Tests :func:`process.get_device_type`.
    """

    def test(self):
        assert process.get_device_type() in ['cpu', 'mpu', 'cuda']


class TestFixRandomSeed:
    """Tests :func:`process.fix_random_seed`.
    """

    def test(self):
        """Tests that no errors are raised.

        *   The random number seed is fixed.
        """
        # random
        process.fix_random_seed(seed=0)
        before = random.randint(0, 10)
        process.fix_random_seed(seed=0)
        after = random.randint(0, 10)
        print(f'{before=}, {after=}')
        assert before == after

        # np.random
        process.fix_random_seed(seed=0)
        before = np.random.randint(0, 10)
        process.fix_random_seed(seed=0)
        after = np.random.randint(0, 10)
        print(f'{before=}, {after=}')
        assert before == after

        # to.rand
        process.fix_random_seed(seed=0)
        before = to.rand(size=[1])
        process.fix_random_seed(seed=0)
        after = to.rand(size=[1])
        print(f'{before=}, {after=}')
        assert before == after


class TestSetWeight:
    """Tests :func:`process.set_weight`.
    """
    params = {
        K.RESULT: DATA_RESULT_DPATH,
        K.MODEL: {K.KIND: 'simple'},
        K.LAYER: {
            K.KIND: ['flatten', 'linear_1', 'relu', 'linear_2'],
            'flatten': Layer.FLATTEN,
            'linear_1': Layer.LINEAR_1,
            'linear_2': Layer.LINEAR_2,
            'relu': Layer.RELU,
        },
        K.DEVICE: 'cpu',
    }

    def test(self):
        """Tests that no errors are raised.

        *   The initialization weights are replaced by the trained weights.
        """
        process.fix_random_seed(seed=0)
        model: to.nn.Module = SetupModel(params=self.params).setup()

        process.fix_random_seed(seed=0)
        weighted_model = SetupModel(params=self.params).setup()
        weighted_model = process.set_weight(params=self.params, model=weighted_model)

        for m, m_w in zip(model.named_parameters(), weighted_model.named_parameters()):
            print(f'{m=}')
            print(f'{m_w=}')
            print(f'{np.where(m[1].detach().numpy() != m_w[1].detach().numpy())=}')
            assert m[0] == m_w[0]
            assert (m[1].detach().numpy() != m_w[1].detach().numpy()).any()


class TestRecursiveReplace:
    """Tests :func:`process.recursive_replace`.
    """
    params = {
        'aaa': None,
        'bbb': {
            'ccc': (None, None),
        },
        'ddd': {
            'eee': [None],
            'fff': {
                'ggg': None,
            },
        },
    }

    def test(self):
        """Tests that no errors are raised.

        *   The replaced value must match the inverse replaced value.
        """
        data = process.recursive_replace(data=self.params, fm_val=None, to_val='None')
        print(f'{data=}')
        data = process.recursive_replace(data=self.params, fm_val='None', to_val=None)
        assert self.params == data


class TestParseTarFnameNumber:
    """Tests :func:`process.parse_tar_fname_number`
    """

    def test(self):
        dpath = Path('data', 'mnist', 'train')
        assert process.parse_tar_fname_number(dpath=dpath) == '{00000..00005}.tar'
