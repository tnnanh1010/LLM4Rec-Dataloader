import logging
import os
import sys

def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")
def get_checkpoint(directory, ckpt_name):
    ckpt_path = os.path.join(directory, ckpt_name)
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        return None

import csv
import json
from logging import Logger
import os
from typing import List


def reset_logger(logger: Logger):
    r"""
    Remove all the filters and the handlers from this logger

    Args:
        logger: a logger.

    Returns:
        None
    """
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilter(f)


def log_csv(path: str, data: List[object]):
    r"""
    Append a new line to a CSV log file already exists

    Args:
        path: path to CSV file.
        data: newline data.

    Returns:
        None
    """
    with open(path, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def create_csv(path: str, header: List[str] = None):
    r"""
    Create a new CSV file

    Args:
        path: path to CSV file.
        header: the header of the CSV file.

    Returns:
        None
    """
    if not os.path.exists(path):
        with open(path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            if header is not None:
                writer.writerow(header)


def log_json(path: str, data: dict, name: str):
    r"""
    Log data to a JSON file

    Args:
        path: path to the JSON file.
        data: data to record.
        name: name of JSON file.

    Returns:
        None
    """
    json_path = os.path.join(path, f'{name}.json')
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(obj=vars(data), fp=f, ensure_ascii=False, indent=4, sort_keys=True)