"""This is the module that loads Cifar data.
"""

from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, override

from lib.common.define import ParamKey, ParamLog
from lib.common.process import parse_tar_fname_number
from lib.data.base import BaseLoadData

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


class Cifar(BaseLoadData):
    """Loads Cifar data.

    Args:
        params (dict[str, Any]): parameters.
    """
    #: ClassVar[list[int]]: image size. (vertical x horizontal x channel)
    size: ClassVar[list[int]] = [32, 32, 3]
    #: int: all number of data.
    n_data: int
    #: ClassVar[list[int]]: input shape. (after preprocess)
    input_shape_model: ClassVar[list[int]] = size.copy()
    #: ClassVar[list[int]]: label shape. (after preprocess)
    label_shape_model: ClassVar[list[int]] = [10] # cifa10: [10], cifar100: [100]

    def __init__(self, params: dict[str, Any]) -> None:
        dpath = Path(params[K.DPATH])
        file_pattern = parse_tar_fname_number(dpath=dpath)
        params[K.FILE_PATTERN] = Path(dpath, file_pattern).as_posix()
        if dpath.stem == 'train':
            self.n_data = 50000
        elif dpath.stem == 'test':
            self.n_data = 10000
        else:
            LOGGER.error(
                f'the dataset path must be "train/%05d.tar" or "test/%05d.tar".',
            )
            raise ValueError
        if params[K.DATA][K.KIND] == 'cifar100':
            self.label_shape_model = [100]
        super().__init__(params=params)

    @override
    def set_model_il_shape(self) -> None:
        """Sets the shape of the preprocessed inputs and labels.

        This method is decorated by ``@override``.
        """
