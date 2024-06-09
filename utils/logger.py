import logging
from pathlib import Path


def get_logger(
        namespace: str,
        output_path: str,
        console: bool = True,
        level: str = "INFO",
        fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Get logger with name, save_dir and save_log.

    Args:
        namespace (str): Name of the logger.
        output_path (str): Output path to save log.
        console (bool): Enable console logging. Default is `True`.
        level (str): Logging level. Default is `INFO`.
        fmt (str): Logging format. Default is `%(asctime)s - %(name)s - %(levelname)s - %(message)s`.

    Returns:
        logging.Logger: Logger.

    """
    if not Path(output_path).parent.exists():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(namespace)
    logger.setLevel(logging.getLevelName(level))
    formatter = logging.Formatter(fmt)
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.getLevelName(level))
        ch.setFormatter(formatter)
        ch.flush()
        logger.addHandler(ch)
    fh = logging.FileHandler(output_path)
    fh.setLevel(logging.getLevelName(level))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
