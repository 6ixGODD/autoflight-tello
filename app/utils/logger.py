import datetime
import logging
from pathlib import Path


def get_logger(name, save_dir, enable_ch=True):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    ch.flush()
    # ISO 8601 Time format
    fh = logging.FileHandler(Path(
        save_dir,
        f"{name}_{datetime.datetime.now().strftime('%Y-%m-%dT%H%M%SZ')}.log"
    ))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    if enable_ch:
        logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info("Log file: {}".format(
        Path(
            save_dir,
            f"{name}_{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}.log"
        )
    ))
    return logger
