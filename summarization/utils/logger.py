import logging
import sys


def get_logger(name):
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    return log
