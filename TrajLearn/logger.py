import os
import logging
from logging.handlers import RotatingFileHandler

def get_logger(log_directory,
               name,
               phase="train",
               console_level=logging.DEBUG,
               file_level=logging.INFO,
               max_file_size=10 * 1024 * 1024,
               backup_count=5,
               log_format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s',
               date_format='%Y-%m-%d %H:%M:%S'):
    """
    Creates and configures a logger with console and file handlers.

    Parameters:
    - log_directory (str): Directory where log files will be stored.
    - phase (str): Logging phase (e.g., "train", "test"). The log file will be named accordingly.
    - console_level (int): Logging level for console output (default: logging.DEBUG).
    - file_level (int): Logging level for file output (default: logging.INFO).
    - max_file_size (int): Maximum size of the log file before it gets rotated (default: 10MB).
    - backup_count (int): Number of backup log files to keep (default: 5).
    - log_format (str): Format of the log messages.
    - date_format (str): Format of the date in log messages.

    Returns:
    - logger (logging.Logger): Configured logger object.
    """
    logger = logging.getLogger(f"{name}-{phase}")

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(log_format, date_format)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)

        os.makedirs(log_directory, exist_ok=True)
        logger.log_directory = log_directory

        logfile = os.path.join(log_directory, f"{phase}.log")
        file_handler = RotatingFileHandler(logfile, mode='a',
                                           maxBytes=max_file_size, backupCount=backup_count)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
