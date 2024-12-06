"""This is the module that defines the configuration.
"""

import zoneinfo
from dataclasses import dataclass
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING
from typing import ClassVar

#: ZoneInfo class.
ZoneInfo = zoneinfo.ZoneInfo(key='Asia/Tokyo')


@dataclass
class ParamFileName:
    """Defines the file name.
    """
    LOSS: str = 'metrics.csv'
    WIGHT: str = '{epoch:03d}_{loss:.3f}_{val_loss:.3f}'


@dataclass
class ParamKey:
    """Defines the dictionary key for the main parameters.
    """
    LEVEL: str = 'level'
    SEED: str = 'seed'
    PARAM: str = 'param'
    RESULT: str = 'result'
    TRAIN: str = 'train'
    VALID: str = 'valid'
    EVAL: str = 'eval'
    BATCH: str = 'batch'
    BATCH_TRAIN: str = 'batch_train'
    BATCH_VALID: str = 'batch_valid'
    SHUFFLE: str = 'shuffle'
    NUM_WORKERS: str = 'num_workers'
    EPOCHS: str = 'epochs'

    DATA: str = 'data'
    PROCESS: str = 'process'
    MODEL: str = 'model'
    LAYER: str = 'layer'
    OPT: str = 'opt'
    LOSS: str = 'loss'
    METRICS: str = 'metrics'
    CB: str = 'cb'
    KIND: str = 'kind'

    DPATH: str = 'dpath'
    FILE_PATTERN: str = 'file_pattern'
    DEVICE: str = 'device'
    MODEL_PARAMS: str = 'model_params'


@dataclass
class ParamLog:
    """Defines the parameters used in the logging configuration.

    This function is decorated by ``@dataclass``.
    """
    SH: str = 'sh'
    FH: str = 'fh'

    #: str: The name to pass to ``logging.getLogger``.
    NAME: str = 'main'
    #: ClassVar[dict[str, int]]: Log level.
    #:
    #: *    key=sh: stream handler.
    #: *    key=fh: file handler.
    LEVEL: ClassVar[dict[str, int]] = {
        SH: DEBUG,
        FH: DEBUG,
    }
    #: str: File path.
    FPATH: str = 'log/log.txt'
    #: int: Max file size.
    SIZE: int = int(1e+6)
    #: int: Number of files.
    NUM: int = 10
