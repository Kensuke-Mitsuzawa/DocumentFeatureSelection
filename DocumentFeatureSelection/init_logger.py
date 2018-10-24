import logging
import sys
from logging import Formatter, StreamHandler

# Formatter
custmoFormatter = Formatter(
    fmt='[%(asctime)s]%(levelname)s - %(filename)s#%(funcName)s:%(lineno)d: %(message)s',
    datefmt='Y/%m/%d %H:%M:%S'
)

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(custmoFormatter)

LOGGER_NAME = 'DocumentFeatureSelection'
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False
