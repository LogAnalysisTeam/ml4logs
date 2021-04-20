# ===== IMPORTS =====
# === Standard library ===
import logging


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def setup_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    LOG_FORMAT = '[{asctime}][{levelname}][{name}] {message}'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(
        fmt=LOG_FORMAT, datefmt=DATE_FORMAT, style='{')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    logger.info('Root logger is configured')


def count_file_lines(path):
    logger.info('Count lines in \'%s\'', path)
    with path.open(encoding='utf8') as in_f:
        for i, _ in enumerate(in_f):
            pass
    return i + 1


def mkdirs(files=[], folders=[]):
    for file_ in files:
        file_.parent.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
