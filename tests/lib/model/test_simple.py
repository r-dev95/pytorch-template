"""This is the module that tests simple.py.
"""

import sys
from logging import ERROR, INFO, getLogger

import pytest
import torch as to
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_pytorch/')
from template_pytorch.lib.common.define import ParamKey, ParamLog
from template_pytorch.lib.model import simple

sys.path.append('../tests')
from define import Layer

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCheckParams:
    """Tests :func:`base.check_params`.
    """
    params = {K.DEVICE: 'cpu'}
    params_raise = {}

    all_log = []
    for key in [K.DEVICE]:
        all_log.append(('main', ERROR  , f'The key "{key}" for variable "params" is missing.'))

    def test(self):
        """Tests that no errors are raised.
        """
        simple.check_params(params=self.params)

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        caplog.set_level(INFO)
        with pytest.raises(ValueError):
            simple.check_params(params=self.params_raise)

        assert caplog.record_tuples == self.all_log


class TestSimpleModel:
    """Tests :class:`simple.SimpleModel`.
    """
    params = {
        K.LAYER: {
            K.KIND: ['linear'],
            'linear': Layer.LINEAR_0,
        },
        K.DEVICE: 'cpu',
    }

    def test(self):
        """Tests that no errors are raised.

        *   The model makes predictions as expected.
        """
        weight = 10

        model = simple.SimpleModel(params=self.params)
        weights = model.state_dict()
        weights['model_layers.0.weight'] = to.Tensor([[weight]])
        weights['model_layers.0.bias'] = to.Tensor([0])
        model.load_state_dict(weights)
        weights = list(model.named_parameters())

        inputs = to.Tensor([[1], [2], [3]])
        preds = model(inputs)
        print(f'{preds=}, {weights=}')

        assert (inputs.detach().numpy() * weight == preds.detach().numpy()).all()
