import datetime
import logging
from pathlib import Path


def get_logger(name, save_dir, enable_ch=True):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    ch.flush()
    fh = logging.FileHandler(Path(save_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    if enable_ch:
        logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info("# log file: {}".format(
        Path(save_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"))
    )
    return logger
