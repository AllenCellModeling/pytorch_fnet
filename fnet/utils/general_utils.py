from pathlib import Path
from typing import Callable, List, Optional, Sequence
import importlib
import inspect
import logging
import os
import sys
import time

import pandas as pd


logger = logging.getLogger(__name__)


def files_from_dir(
    path_dir: str, extensions: Optional[Sequence[str]] = None
) -> List[str]:
    """Returns sorted list of files in a directory with optional extension(s).

    Parameters
    ----------
    path_dir
        Input directory.
    extensions
        Optional file extensions.

    """
    if extensions is None:
        extensions = [""]  # Allows for all extensions
    paths = []
    for entry in os.scandir(path_dir):
        if any(entry.path.endswith(ext) for ext in extensions):
            paths.append(entry.path)
    return sorted(paths)


def str_to_object(str_o: str):
    """Get object from string.

    Parameters
    ----------
    str_o
        Fully qualified object name.

    """
    parts = str_o.split(".")
    if len(parts) > 1:
        module = importlib.import_module(".".join(parts[:-1]))
        return getattr(module, parts[-1])
    return inspect.currentframe().f_back.f_globals[str_o]


def to_objects(slist):
    """Get a list of objects from list of object __repr__s."""
    if slist is None:
        return None
    olist = list()
    for s in slist:
        if not isinstance(s, str):
            if s is None:
                continue
            olist.append(s)
            continue
        if s.lower() == "none":
            continue
        s_split = s.split(".")
        for idx_part, part in enumerate(s_split):
            if not part.isidentifier():
                break
        importee = ".".join(s_split[:idx_part])
        so = ".".join(s_split[idx_part:])
        if len(importee) > 0:
            module = importlib.import_module(importee)  # noqa: F841
            so = "module." + so
        olist.append(eval(so))
    return olist


def retry_if_oserror(fn: Callable):
    """Retries input function if an OSError is encountered."""

    def wrapper(*args, **kwargs):
        count = 0
        while True:
            count += 1
            try:
                fn(*args, **kwargs)
                break
            except OSError as err:
                wait = 2 ** min(count, 5)
                logger.info(f"Attempt {count} failed: {err}. Waiting {wait} seconds.")
                time.sleep(wait)

    return wrapper


def get_args():
    """Returns the arguments passed to the calling function.

    Example:

    >>> def foo(a, b, *args, **kwargs):
    ...     print(get_args())
    ...
    >>> foo(1, 2, 3, 'bar', fizz='buzz')
    ({'b': 2, 'a': 1, 'fizz': 'buzz'}, (3, 'bar'))

    References:
    kbyanc.blogspot.com/2007/07/python-aggregating-function-arguments.html

    Returns
    -------
    dict
         Named arguments
    list
         Unnamed positional arguments

    """
    frame = inspect.stack()[1].frame  # Look at caller
    _, varargs, kwargs, named_args = inspect.getargvalues(frame)
    named_args = dict(named_args)
    named_args.update(named_args.pop(kwargs, []))
    pos_args = named_args.pop(varargs, [])
    return named_args, pos_args


def str_to_class(string: str):
    """Return class from string representation."""
    idx_dot = string.rfind(".")
    if idx_dot < 0:
        module_str = "fnet.nn_modules"
        class_str = string
    else:
        module_str = string[:idx_dot]
        class_str = string[idx_dot + 1 :]
    module = importlib.import_module(module_str)
    return getattr(module, class_str)


def add_augmentations(df: pd.DataFrame) -> pd.DataFrame:
    """Adds augmented versions of dataframe rows.

    This is intended to be used on dataframes that represent datasets. Two
    columns will be added: flip_y, flip_x. Each dataframe row will be
    replicated 3 more times with flip_y, flip_x, or both columns set to 1.

    Parameters
    ----------
    df
        Dataset dataframe to be augmented.

    Returns
    -------
    pd.DataFrame
        Augmented dataset dataframe.

    """
    df_flip_y = df.assign(flip_y=1)
    df_flip_x = df.assign(flip_x=1)
    df_both = df.assign(flip_y=1, flip_x=1)
    name_index = df.index.name
    df_aug = pd.concat(
        [df, df_flip_y, df_flip_x, df_both], ignore_index=True, sort=False
    ).rename_axis(name_index)
    return df_aug


def whats_my_name(obj: object):
    """Returns object's name."""
    return obj.__module__ + "." + obj.__qualname__


def create_formatter():
    """Creates a default logging Formatter."""
    return logging.Formatter("%(levelname)s:%(name)s: %(message)s")


def add_logging_file_handler(path_save: Path) -> None:
    """Adds a file handler to fnet logger.

    Parameters
    ----------
    path_save
        Location to save logging records.

    Returns
    -------
    None

    """
    path_save.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path_save, mode="a")
    fh.setFormatter(create_formatter())
    logging.getLogger("fnet").addHandler(fh)


def init_fnet_logging() -> None:
    """Initializes logging for fnet.

    Parameters
    ----------
    path_save
        Location to save logging records.

    Returns
    -------
    None

    """
    # Remove root logger handlers potentially set by third-party packages
    logger_root = logging.getLogger()
    for handler in logger_root.handlers:
        logger_root.removeHandler(handler)
    # Init fnet logger
    logger_fnet = logging.getLogger("fnet")
    logger_fnet.setLevel(logging.INFO)
    if logger_fnet.hasHandlers():  # avoids redundant handlers
        return
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(create_formatter())
    logger_fnet.addHandler(sh)
