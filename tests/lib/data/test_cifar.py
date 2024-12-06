"""This is the module that tests cifar.py.
"""

import sys
from logging import getLogger
from pathlib import Path

sys.path.append('../template_pytorch/')
from template_pytorch.lib.common.define import ParamKey, ParamLog
from template_pytorch.lib.data import cifar

sys.path.append('../tests')
from define import DATA_PARENT_DPATH

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCifar:
    """Tests :class:`cifar.Cifar`.
    """
    params = {
        K.RESULT: DATA_PARENT_DPATH,
        K.PROCESS: {
            K.KIND: [],
        },
        K.BATCH: 1000,
        K.SHUFFLE: None,
        K.DEVICE: 'cpu',
        K.NUM_WORKERS: 4,
    }

    def _func(self, kind: str, phase: str):
        self.params[K.DATA] = {K.KIND: kind}
        self.params[K.DPATH] = Path(self.params[K.RESULT], kind, phase)
        data = cifar.Cifar(params=self.params)
        loader = data.make_loader_example()
        for inputs, labels in loader:
            print(f'\r{kind}-{phase}: {inputs.numpy().shape=}, {labels.numpy().shape=}')

    def test(self):
        """Tests that no errors are raised.

        *   Data can load correctly. (cifar10, cifar100)
        """
        # cifar10
        self._func(kind='cifar10', phase='train')
        self._func(kind='cifar10', phase='test')
        # cifar100
        self._func(kind='cifar100', phase='train')
        self._func(kind='cifar100', phase='test')
