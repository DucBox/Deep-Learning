import logging

def setup_logger(log_file="project.log"):
    """
    Set up a logger to log messages to a file and console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        Logger: Configured logger instance.
    """
    logger = logging.getLogger("project_logger")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
import logging

def setup_logger(log_file="project.log"):
    """
    Set up a logger to log messages to a file and console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        Logger: Configured logger instance.
    """
    logger = logging.getLogger("project_logger")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
