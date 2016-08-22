LOGGER_NAME = 'DocumentFeatureSelection'

import logging
import os
import sys
from logging import getLogger, Formatter, Logger, StreamHandler
from logging.handlers import SMTPHandler, RotatingFileHandler, TimedRotatingFileHandler

# Formatter
custmoFormatter = Formatter(
    fmt='[%(asctime)s]%(levelname)s - %(filename)s#%(funcName)s:%(lineno)d: %(message)s',
    datefmt='Y/%m/%d %H:%M:%S'
)

# StreamHandler
STREAM_LEVEL = logging.DEBUG
STREAM_FORMATTER = custmoFormatter
STREAM = sys.stdout

st_handler = StreamHandler(stream=STREAM)
st_handler.setLevel(STREAM_LEVEL)
st_handler.setFormatter(STREAM_FORMATTER)


def init_logger(logger:logging.Logger)->logging.Logger:
    logger.addHandler(st_handler)
    logger.propagate = False

    return logger