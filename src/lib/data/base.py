"""This is the module load data.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any, ClassVar

import torch as to
import webdataset as wds

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.data.processor import Processor

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any]) -> None:
    """Checks the :class:`BaseLoadData` parameters.

    Args:
        params (dict[str, Any]): parameters.
    """
    error = False # error: True
    keys = [K.DEVICE, K.FILE_PATTERN, K.BATCH, K.SHUFFLE, K.NUM_WORKERS]
    for key in keys:
        if key not in params:
            error = True
            LOGGER.error(f'The key "{key}" for variable "params" is missing.')
    if error:
        raise ValueError


class BaseLoadData:
    """Loads data.

    *   Make a data pipeline to load a shard data (webdataset).

    Args:
        params (dict[str, Any]): parameters.

    .. attention::

        Child classes that inherit this class must set the pattern of file paths to
        ``params[K.FILE_PATTERN]`` before running ``super().__init__(params=params)``.
    """
    #: int: all number of data.
    n_data: int
    #: ClassVar[list[int]]: input shape. (after preprocess)
    input_shape_model: ClassVar[list[int]]
    #: ClassVar[list[int]]: label shape. (after preprocess)
    label_shape_model: ClassVar[list[int]]

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        check_params(params=params)

        self.Processor = Processor(params=params)
        self.set_model_il_shape()

        self.steps_per_epoch = (self.n_data - 1) // params[K.BATCH] + 1

    def set_model_il_shape(self) -> None:
        """Sets the shape of the preprocessed inputs and labels.
        """
        raise NotImplementedError

    def process(self, data: tuple[to.Tensor, to.Tensor]) -> tuple[to.Tensor, to.Tensor]:
        """Runs process data.

        *   Run :meth:`lib.data.processor.Processor.run`.

        Args:
            data (tuple[to.Tensor, to.Tensor]): tuple of inputs and labels.

        Returns:
            to.Tensor: input. (after process)
            to.Tensor: label. (after process)
        """
        x, y = data
        x, y = to.Tensor(x), to.Tensor(y)
        if self.params[K.PROCESS][K.KIND] is not None:
            x, y = self.Processor.run(x=x, y=y)
        x = x.to(dtype=to.float, device=self.params[K.DEVICE])
        y = y.to(dtype=to.float, device=self.params[K.DEVICE])
        # x = to.reshape(x, self.input_shape_model)
        # y = to.reshape(y, self.label_shape_model)
        return x, y

    def make_loader_example(self, seed: int = 0) -> Callable:
        """Makes data loader.

        #.  Set the file path pattern, random seed, and shuffle flag for sharded data.
            (``wds.WebDataset``)
        #.  Set the decoding configuration.
            (``wds.WebDataset.decode``)
        #.  Set the file type.
            (``wds.WebDataset.to_tuple``)
        #.  Set the preprocess function.
            (``wds.WebDataset.map``)
        #.  Set the shuffle configuration.
            (``wds.WebDataset.shuffle``)
        #.  Set the dataset (``wds.WebDataset``), batch size, and number of workers.
            (``to.utils.data.DataLoader``)

        Args:
            seed (int): random seed.

        Returns:
            Callable: data pipeline. (``to.utils.data.DataLoader``)
        """
        dataset = wds.WebDataset(
            urls=self.params[K.FILE_PATTERN],
            seed=seed,
            shardshuffle=True,
            empty_check=False,
        )
        dataset: wds.WebDataset = dataset.decode()
        dataset: wds.WebDataset = dataset.to_tuple('inputs.npy', 'labels.npy')
        dataset: wds.WebDataset = dataset.map(f=self.process)
        if self.params[K.SHUFFLE] is not None:
            dataset: wds.WebDataset = dataset.shuffle(size=self.params[K.SHUFFLE])
        dataset = to.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.params[K.BATCH],
            num_workers=self.params[K.NUM_WORKERS],
        )

        return dataset
