import logging

import config


def init_logging(LOG_FILE = None):
    if LOG_FILE is None:
        LOG_FILE = config.LOG_FILE

    logging.basicConfig(level=logging.DEBUG,
                        format=config.LOG_FORMAT,
                        filename=LOG_FILE,
                        filemode='a')

    # Logging handler to output to the console when running
    ch = logging.StreamHandler()
    ch.setLevel(config.LOG_CONSOLE_LEVEL)

    # Logging handler to output to the log file
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(config.LOG_FILE_LEVEL)

    formatter = logging.Formatter(config.LOG_FORMAT)
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger = logging.getLogger('')
    for h in logger.handlers:
        logger.removeHandler(h)

    logger.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.addHandler(fh)
