"""This is the module that defines the test configuration.
"""

from dataclasses import dataclass
from logging import getLogger

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)

DATA_PARENT_DPATH = 'data'
DATA_RESULT_DPATH = f'{DATA_PARENT_DPATH}/result'

# -----------------------------------------------
# processor parameters
# -----------------------------------------------
@dataclass
class Proc:
    ONE_HOT = {
        'num_classes': 10,
    }
    RESCALE = {
        'scale': 10,
        'offset': 0,
    }
# -----------------------------------------------
# model layer parameters
# -----------------------------------------------
@dataclass
class Layer:
    FLATTEN = {
        'start_dim': 1,
        'end_dim': -1,
    }
    LINEAR_0 = {
        'in_features': 1,
        'out_features': 1,
        'bias': True,
    }
    LINEAR_1 = {
        'in_features': 784,
        'out_features': 100,
        'bias': True,
    }
    LINEAR_2 = {
        'in_features': 100,
        'out_features': 10,
        'bias': True,
    }
    LINEAR_3 = {
        'in_features': 1568,
        'out_features': 100,
        'bias': True,
    }
    CONV2D_1 = {
        'in_channels': 1,
        'out_channels': 8,
        'kernel_size': 3,
        'stride': 1,
        'padding': 0,
        'dilation': 1,
        'groups': 1,
        'bias': True,
        'padding_mode': 'zeros',
    }
    CONV2D_2 = {
        'in_channels': 8,
        'out_channels': 8,
        'kernel_size': 3,
        'stride': 1,
        'padding': 0,
        'dilation': 1,
        'groups': 1,
        'bias': True,
        'padding_mode': 'zeros',
    }
    MAXPOOL2D = {
        'kernel_size': 2,
        'stride': 2,
        'padding': 0,
        'dilation': 1,
        'return_indices': False,
        'ceil_mode': False,
    }
    RELU = {
        'inplace': False,
    }
# -----------------------------------------------
# optimizer method parameters
# -----------------------------------------------
@dataclass
class Opt:
    SGD = {
        # 'params': None,
        'lr': 0.001,
        'momentum': 0,
        'dampening': 0,
        'weight_decay': 0,
        'nesterov': False,
        'maximize': False,
        'foreach': None,
        'differentiable': False,
        'fused': None,
    }
    ADAM = {
        # 'params': None,
        'lr': 0.001,
        'betas': [0.9, 0.999],
        'eps': 0.00000001,
        'weight_decay': 0,
        'amsgrad': False,
        'foreach': None,
        'maximize': False,
        'capturable': False,
        'differentiable': False,
        'fused': None,
    }
# -----------------------------------------------
# loss function parameters
# -----------------------------------------------
@dataclass
class Loss:
    MSE = {
        'reduction': 'sum_over_batch_size',
    }
    CE = {
        'weight': None,
        'ignore_index': -100,
        'reduction': 'mean',
        'label_smoothing': 0,
    }
# -----------------------------------------------
# metrics parameters
# -----------------------------------------------
@dataclass
class Metrics:
    MSE = {
        'squared': True,
        'num_outputs': 1,
    }
    BACC = {
        'threshold': 0.5,
        'multidim_average': 'global',
        'ignore_index': None,
        'validate_args': True,
    }
    MACC = {
        'num_classes': 10,
        'top_k': 1,
        'average': 'macro',
        'multidim_average': 'global' ,
        'ignore_index': None,
        'validate_args': True,
    }
# -----------------------------------------------
# callback parameters
# -----------------------------------------------
@dataclass
class CB:
    MS = {
        'max_depth': 3,
    }
    MCP = {
        # 'dirpath':,
        # 'filename':,
        'monitor': None,
        'verbose': False,
        'save_last': None,
        'save_top_k': -1,
        'save_weights_only': False,
        'mode': 'min',
        'auto_insert_metric_name': True,
        'every_n_train_steps': None,
        'train_time_interval': None,
        'every_n_epochs': 1,
        'save_on_train_epoch_end': None,
        'enable_version_counter': True,
    }
