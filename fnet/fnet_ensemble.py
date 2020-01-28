from typing import Union, List
import logging
import os

import numpy as np
import torch

from fnet.fnet_model import Model
from fnet.utils.general_utils import str_to_class


logger = logging.info(__name__)


def _load_model(path_model: str) -> Model:
    """Load saved model from path."""
    state = torch.load(path_model)
    fnet_model_class = state["fnet_model_class"]
    fnet_model_kwargs = state["fnet_model_kwargs"]
    model = str_to_class(fnet_model_class)(**fnet_model_kwargs)
    model.load_state(state, no_optim=True)
    return model


class FnetEnsemble(Model):
    """Ensemble of FnetModels.

    Parameters
    ----------
    paths_model
        Path to a directory of saved models or a list of paths to saved models.

    Attributes
    ----------
    paths_model : Union[str, List[str]]
        Paths to saved models in the ensemble.
    gpu_ids : List[int]
        GPU(s) used for prediction tasks.

    """

    def __init__(self, paths_model: Union[str, List[str]]) -> None:
        if isinstance(paths_model, str):
            assert os.path.isdir(paths_model)
            paths_model = sorted(
                [
                    p.path
                    for p in os.scandir(os.path.abspath(paths_model))
                    if p.path.lower().endswith(".p")
                ]
            )
        assert len(paths_model) > 0
        self.paths_model = paths_model
        self.gpu_ids = -1

    def __str__(self):
        str_out = []
        str_out.append(f"{len(self.paths_model)}-model ensemble:")
        str_out.extend([p for p in self.paths_model])
        return os.linesep.join(str_out)

    def to_gpu(self, gpu_ids: Union[int, list]) -> None:
        """Move network to specified GPU(s).

        Parameters
        ----------
        gpu_ids
            GPU(s) on which to perform training or prediction.

        """
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        self.gpu_ids = gpu_ids

    def predict(
        self, x: Union[torch.Tensor, np.ndarray], tta: bool = False
    ) -> torch.Tensor:
        """Performs model prediction.

        Parameters
        ----------
        x
            Batched input.
        tta
            Set to to use test-time augmentation.

        Returns
        -------
        torch.Tensor
            Model prediction.

        """
        y_hat_mean = None
        for path_model in self.paths_model:
            model = _load_model(path_model)
            model.to_gpu(self.gpu_ids)
            y_hat = model.predict(x=x, tta=tta)
            if y_hat_mean is None:
                y_hat_mean = torch.zeros(*y_hat.size())
            y_hat_mean += y_hat
        return y_hat_mean / len(self.paths_model)

    # Override
    def save(self, path_save: str):
        """Saves model to disk.

        Parameters
        ----------
        path_save
            Filename to which model is saved.

        """
        state = {
            "fnet_model_class": (self.__module__ + "." + self.__class__.__qualname__),
            "fnet_model_kwargs": {"paths_model": self.paths_model},
        }
        dirname = os.path.dirname(path_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(state, path_save)
        logger.info(f"Ensemble model saved to: {path_save}")

    # Override
    def load_state(self, state: dict, no_optim: bool = False):
        return
