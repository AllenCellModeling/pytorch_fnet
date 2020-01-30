from fnet import models
from fnet.fnetlogger import FnetLogger

__author__ = "Gregory R. Johnson"
__email__ = "gregj@alleninstitute.org"
__version__ = "0.2.0"


def get_module_version():
    return __version__


__all__ = ["models", "FnetLogger"]
