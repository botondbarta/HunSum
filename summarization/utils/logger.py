import logging
import sys


def get_logger(name, log_file):
    formatter = logging.Formatter('%(asctime)s %(levelname)-9s %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.addHandler(file_handler)
    return log
