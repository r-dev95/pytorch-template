"""This is the module that tests file.py.
"""

import shutil
from logging import getLogger
from pathlib import Path

import pytest

from lib.common import file
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestJson:
    """Tests :func:`file.dump_json` and :func:`file.load_json`.
    """
    params = {
        K.SEED: 0,
        K.RESULT: 'result',
        K.TRAIN: None,
        K.CB: {
            K.KIND: ['mcp'],
            'mcp': {
                # 'filepath': 'epoch{epoch:03d}_loss{loss:.3f}_{val_loss:.3f}.weights.h5',
                'monitor': 'val_loss',
                'verbose': 0,
                'save_best_only': False,
                'save_weights_only': True,
                'mode': 'auto',
                'save_freq': 'epoch',
                'initial_value_threshold': None,
            },
        },
    }

    @pytest.fixture(scope='class')
    def proc(self):
        dpath = Path(self.params[K.RESULT])
        dpath.mkdir(parents=True, exist_ok=True)
        yield
        shutil.rmtree(dpath)

    def test(self, proc):
        """Tests that no errors are raised.

        *   The data written matches the data read.
        """
        fpath = Path(self.params[K.RESULT], 'log.json')
        file.dump_json(data=self.params, fpath=fpath)
        data = file.load_json(fpath=fpath)
        assert self.params == data


class TestYaml:
    """Tests :func:`file.dump_yaml` and :func:`file.load_yaml`.
    """
    params = {
        K.SEED.value: 0,
        K.RESULT.value: 'result',
        K.TRAIN.value: None,
        K.CB.value: {
            K.KIND.value: ['mcp'],
            'mcp': {
                # 'filepath': 'epoch{epoch:03d}_loss{loss:.3f}_{val_loss:.3f}.weights.h5',
                'monitor': 'val_loss',
                'verbose': 0,
                'save_best_only': False,
                'save_weights_only': True,
                'mode': 'auto',
                'save_freq': 'epoch',
                'initial_value_threshold': None,
            },
        },
    }

    @pytest.fixture(scope='class')
    def proc(self):
        dpath = Path(self.params[K.RESULT])
        dpath.mkdir(parents=True, exist_ok=True)
        yield
        shutil.rmtree(dpath)

    def test(self, proc):
        """Tests that no errors are raised.

        *   The data written matches the data read.
        """
        fpath = Path(self.params[K.RESULT], 'log.yaml')
        file.dump_yaml(data=self.params, fpath=fpath)
        print(f'{self.params=}')
        data = file.load_yaml(fpath=fpath)
        assert self.params == data


class TestToml:
    """Tests :func:`file.dump_toml` and :func:`file.load_toml`.
    """
    params = {
        K.SEED: 0,
        K.RESULT: 'result',
        K.TRAIN: None,
        K.CB: {
            K.KIND: ['mcp'],
            'mcp': {
                # 'filepath': 'epoch{epoch:03d}_loss{loss:.3f}_{val_loss:.3f}.weights.h5',
                'monitor': 'val_loss',
                'verbose': 0,
                'save_best_only': False,
                'save_weights_only': True,
                'mode': 'auto',
                'save_freq': 'epoch',
                'initial_value_threshold': None,
            },
        },
    }

    @pytest.fixture(scope='class')
    def proc(self):
        dpath = Path(self.params[K.RESULT])
        dpath.mkdir(parents=True, exist_ok=True)
        yield
        shutil.rmtree(dpath)

    def test(self, proc):
        """Tests that no errors are raised.

        *   The data written matches the data read.
        """
        fpath = Path(self.params[K.RESULT], 'log.toml')
        file.dump_toml(data=self.params, fpath=fpath)
        data = file.load_toml(fpath=fpath)
        assert self.params == data
