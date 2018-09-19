from typing import Callable
import importlib
import inspect
import pdb  # noqa: F401
import time


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
        if s.lower() == 'none':
            continue
        s_split = s.split('.')
        for idx_part, part in enumerate(s_split):
            if not part.isidentifier():
                break
        importee = '.'.join(s_split[:idx_part])
        so = '.'.join(s_split[idx_part:])
        if len(importee) > 0:
            module = importlib.import_module(importee)  # noqa: F841
            so = 'module.' + so
        olist.append(eval(so))
    return olist


def retry_if_oserror(fn: Callable):
    def wrapper(*args, **kwargs):
        count = 0
        while True:
            count += 1
            try:
                fn(*args, **kwargs)
                break
            except OSError as err:
                wait = 2**min(count, 5)
                print(f'Attempt {count}:', err,
                      f'| waiting {wait} seconds....')
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
    idx_dot = string.rfind('.')
    if idx_dot < 0:
        module_str = 'fnet.nn_modules'
        class_str = string
    else:
        module_str = string[:idx_dot]
        class_str = string[idx_dot + 1:]
    module = importlib.import_module(module_str)
    return getattr(module, class_str)
