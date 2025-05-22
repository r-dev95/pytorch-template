"""This is the module that sets the logging configuration.
"""

from logging import Formatter, Logger, StreamHandler
from logging.handlers import RotatingFileHandler

from lib.common.define import ParamKey, ParamLog

K = ParamKey()


class SetLogging:
    """Sets the logging configuration.

    Args:
        logger (Logger): ``logging.Logger``
        param (ParamLog): :class:`lib.common.define.ParamLog`
    """
    #: logging.Formatter: Log format.
    format = Formatter(
        '[%(asctime)s][%(filename)s][%(funcName)s][%(lineno)s]'
        '[%(levelname)s] - %(message)s',
    )

    def __init__(self, logger: Logger, param: ParamLog) -> None:
        self.logger = logger
        self.param = param

        self.set_level()
        self.set_stream_handler()
        self.set_file_handler()

    def set_level(self) -> None:
        """Sets the log level.

        *   Run ``logging.Logger.setLevel``.
        """
        self.logger.setLevel(self.param.LEVEL[self.param.SH])

    def set_stream_handler(self) -> None:
        """Sets the stream handler configuration.

        *   Set the log level.
            (``logging.StreamHandler.setLevel``)
        *   Set the log format.
            (``logging.StreamHandler.setFormatter``)
        *   Add the handler.
            (``logging.Logger.addHandler``)
        """
        sh = StreamHandler()
        sh.setLevel(level=self.param.LEVEL[self.param.SH])
        sh.setFormatter(fmt=self.format)
        self.logger.addHandler(sh)

    def set_file_handler(self) -> None:
        """Sets the file handler configuration.

        *   Set the file path, size, and number.
            (``logging.RotatingFileHandler``)
        *   Set the log level.
            (``logging.RotatingFileHandler.setLevel``)
        *   Set the log format.
            (``logging.RotatingFileHandler.setFormatter``)
        *   Add the handler.
            (``logging.Logger.addHandler``)
        """
        fh = RotatingFileHandler(
            filename=self.param.FPATH,
            maxBytes=self.param.SIZE,
            backupCount=self.param.NUM,
        )
        fh.setLevel(level=self.param.LEVEL[self.param.FH])
        fh.setFormatter(fmt=self.format)
        self.logger.addHandler(fh)
