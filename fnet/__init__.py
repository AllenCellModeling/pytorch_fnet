from fnet import models
from fnet.fnetlogger import FnetLogger

# Clean these up later - GRJ 2020-02-04
from fnet.cli.train_model import train_model as train
from fnet.cli.predict import main as predict

__author__ = "Gregory R. Johnson"
__email__ = "gregj@alleninstitute.org"
__version__ = "0.2.0"


def get_module_version():
    return __version__


__all__ = ["models", "FnetLogger"]
