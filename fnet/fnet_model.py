"""Module to define main fnet model wrapper class."""


from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union
import logging
import math
import os

from scipy.ndimage import zoom
import numpy as np
import tifffile
import torch

from fnet.metrics import corr_coef
from fnet.predict_piecewise import predict_piecewise as _predict_piecewise_fn
from fnet.transforms import flip_y, flip_x, norm_around_center
from fnet.utils.general_utils import get_args, retry_if_oserror, str_to_object
from fnet.utils.model_utils import move_optim


logger = logging.getLogger(__name__)


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith("Conv"):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_per_param_options(module, wd):
    """Returns list of per parameter group options.

    Applies the specified weight decay (wd) to parameters except parameters
    within batch norm layers and bias parameters.
    """
    if wd == 0:
        return module.parameters()
    with_decay = list()
    without_decay = list()
    for idx_m, (name_m, module_sub) in enumerate(module.named_modules()):
        if list(module_sub.named_children()):
            continue  # Skip "container" modules
        if isinstance(module_sub, torch.nn.modules.batchnorm._BatchNorm):
            for param in module_sub.parameters():
                without_decay.append(param)
            continue
        for name_param, param in module_sub.named_parameters():
            if "weight" in name_param:
                with_decay.append(param)
            elif "bias" in name_param:
                without_decay.append(param)
    # Check that no parameters were missed or duplicated
    n_param_module = len(list(module.parameters()))
    n_param_lists = len(with_decay) + len(without_decay)
    n_elem_module = sum([p.numel() for p in module.parameters()])
    n_elem_lists = sum([p.numel() for p in with_decay + without_decay])
    assert n_param_module == n_param_lists
    assert n_elem_module == n_elem_lists
    per_param_options = [
        {"params": with_decay, "weight_decay": wd},
        {"params": without_decay, "weight_decay": 0.0},
    ]
    return per_param_options


class Model:
    """Class that encompasses a pytorch network and its optimizer.

    """

    def __init__(
        self,
        betas=(0.5, 0.999),
        criterion_class="fnet.losses.WeightedMSE",
        init_weights=True,
        lr=0.001,
        nn_class="fnet.nn_modules.fnet_nn_3d.Net",
        nn_kwargs={},
        scheduler=None,
        weight_decay=0,
        gpu_ids=-1,
    ):
        self.betas = betas
        self.criterion = str_to_object(criterion_class)()
        self.gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
        self.init_weights = init_weights
        self.lr = lr
        self.nn_class = nn_class
        self.nn_kwargs = nn_kwargs
        self.scheduler = scheduler
        self.weight_decay = weight_decay

        self.count_iter = 0
        self.device = (
            torch.device("cuda", self.gpu_ids[0])
            if self.gpu_ids[0] >= 0
            else torch.device("cpu")
        )
        self.optimizer = None
        self._init_model()
        self.fnet_model_kwargs, self.fnet_model_posargs = get_args()
        self.fnet_model_kwargs.pop("self")

    def _init_model(self):
        self.net = str_to_object(self.nn_class)(**self.nn_kwargs)
        if self.init_weights:
            self.net.apply(_weights_init)
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(
            get_per_param_options(self.net, wd=self.weight_decay),
            lr=self.lr,
            betas=self.betas,
        )
        if self.scheduler is not None:
            if self.scheduler[0] == "snapshot":
                period = self.scheduler[1]
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lambda x: (
                        0.01
                        + (1 - 0.01)
                        * (0.5 + 0.5 * math.cos(math.pi * (x % period) / period))
                    ),
                )
            elif self.scheduler[0] == "step":
                step_size = self.scheduler[1]
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size
                )
            else:
                raise NotImplementedError

    def __str__(self):
        out_str = [
            f"*** {self.__class__.__name__} ***",
            f"{self.nn_class}(**{self.nn_kwargs})",
            f"iter: {self.count_iter}",
            f"gpu: {self.gpu_ids}",
        ]
        return os.linesep.join(out_str)

    def get_state(self):
        return {
            "fnet_model_class": (self.__module__ + "." + self.__class__.__qualname__),
            "fnet_model_kwargs": self.fnet_model_kwargs,
            "fnet_model_posargs": self.fnet_model_posargs,
            "nn_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "count_iter": self.count_iter,
        }

    def to_gpu(self, gpu_ids: Union[int, List[int]]) -> None:
        """Move network to specified GPU(s).

        Parameters
        ----------
        gpu_ids
            GPU(s) on which to perform training or prediction.

        """
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        self.gpu_ids = gpu_ids
        self.device = (
            torch.device("cuda", self.gpu_ids[0])
            if self.gpu_ids[0] >= 0
            else torch.device("cpu")
        )
        self.net.to(self.device)
        if self.optimizer is not None:
            move_optim(self.optimizer, self.device)

    def save(self, path_save: str):
        """Saves model to disk.

        Parameters
        ----------
        path_save
            Filename to which model is saved.

        """
        dirname = os.path.dirname(path_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            logger.info(f"Created: {dirname}")
        curr_gpu_ids = self.gpu_ids
        self.to_gpu(-1)
        retry_if_oserror(torch.save)(self.get_state(), path_save)
        self.to_gpu(curr_gpu_ids)

    def load_state(self, state: dict, no_optim: bool = False):
        self.count_iter = state["count_iter"]
        self.net.load_state_dict(state["nn_state"])
        if no_optim:
            self.optimizer = None
            return
        self.optimizer.load_state_dict(state["optimizer_state"])

    def train_on_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
    ) -> float:
        """Update model using a batch of inputs and targets.

        Parameters
        ----------
        x_batch
            Batched input.
        y_batch
            Batched target.
        weight_map_batch
            Optional batched weight map.

        Returns
        -------
        float
            Loss as determined by self.criterion.

        """
        if self.scheduler is not None:
            self.scheduler.step()
        self.net.train()
        x_batch = x_batch.to(dtype=torch.float32, device=self.device)
        y_batch = y_batch.to(dtype=torch.float32, device=self.device)
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(self.net, device_ids=self.gpu_ids)
        else:
            module = self.net

        self.optimizer.zero_grad()
        y_hat_batch = module(x_batch)
        args = [y_hat_batch, y_batch]
        if weight_map_batch is not None:
            args.append(weight_map_batch)
        loss = self.criterion(*args)
        loss.backward()
        self.optimizer.step()
        self.count_iter += 1
        return loss.item()

    def _predict_on_batch_tta(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Performs model prediction using test-time augmentation."""
        augs = [None, [flip_y], [flip_x], [flip_y, flip_x]]
        x_batch = x_batch.numpy()
        y_hat_batch_mean = None
        for aug in augs:
            x_batch_aug = x_batch.copy()
            if aug is not None:
                for trans in aug:
                    x_batch_aug = trans(x_batch_aug)
            y_hat_batch = self.predict_on_batch(x_batch_aug.copy()).numpy()
            if aug is not None:
                for trans in aug:
                    y_hat_batch = trans(y_hat_batch)
            if y_hat_batch_mean is None:
                y_hat_batch_mean = np.zeros(y_hat_batch.shape, dtype=np.float32)
            y_hat_batch_mean += y_hat_batch
        y_hat_batch_mean /= len(augs)
        return torch.tensor(
            y_hat_batch_mean, dtype=torch.float32, device=torch.device("cpu")
        )

    def predict_on_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Performs model prediction on a batch of data.

        Parameters
        ----------
        x_batch
            Batch of input data.

        Returns
        -------
        torch.Tensor
            Batch of model predictions.

        """
        x_batch = torch.tensor(x_batch, dtype=torch.float32, device=self.device)

        if len(self.gpu_ids) > 1:
            network = torch.nn.DataParallel(self.net, device_ids=self.gpu_ids)
        else:
            network = self.net

        network.eval()
        with torch.no_grad():
            y_hat_batch = network(x_batch).cpu()

        network.train()

        return y_hat_batch

    def predict(
        self, x: Union[torch.Tensor, np.ndarray], tta: bool = False
    ) -> torch.Tensor:
        """Performs model prediction on a single example.

        Parameters
        ----------
        x
            Input data.
        piecewise
            Set to perform piecewise predictions. i.e., predict on patches of
            the input and stitch together the predictions.
        tta
            Set to use test-time augmentation.

        Returns
        -------
        torch.Tensor
            Model prediction.

        """
        x_batch = torch.unsqueeze(torch.tensor(x), 0)
        if tta:
            return self._predict_on_batch_tta(x_batch).squeeze(0)
        return self.predict_on_batch(x_batch).squeeze(0)

    def predict_piecewise(
        self, x: Union[torch.Tensor, np.ndarray], **predict_kwargs
    ) -> torch.Tensor:
        """Performs model prediction piecewise on a single example.

        Predicts on patches of the input and stitchs together the predictions.

        Parameters
        ----------
        x
            Input data.
        **predict_kwargs
            Kwargs to pass to predict method.

        Returns
        -------
        torch.Tensor
            Model prediction.

        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if len(x.size()) == 4:
            dims_max = [None, 32, 512, 512]
        elif len(x.size()) == 3:
            dims_max = [None, 1024, 1024]
        y_hat = _predict_piecewise_fn(
            self, x, dims_max=dims_max, overlaps=16, **predict_kwargs
        )
        return y_hat

    def test_on_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
    ) -> float:
        """Test model on a batch of inputs and targets.

        Parameters
        ----------
        x_batch
            Batched input.
        y_batch
            Batched target.
        weight_map_batch
            Optional batched weight map.

        Returns
        -------
        float
            Loss as evaluated by self.criterion.

        """

        y_hat_batch = self.predict_on_batch(x_batch)

        args = [y_hat_batch, y_batch]

        if weight_map_batch is not None:
            args.append(weight_map_batch)

        loss = self.criterion(*args)

        return loss.item()

    def test_on_iterator(self, iterator: Iterator, **kwargs: dict) -> float:
        """Test model on iterator which has items to be passed to
        test_on_batch.

        Parameters
        ----------
        iterator
            Iterator that generates items to be passed to test_on_batch.
        kwargs
            Additional keyword arguments to be passed to test_on_batch.

        Returns
        -------
        float
            Mean loss for items in iterable.

        """
        loss_sum = 0
        for item in iterator:
            loss_sum += self.test_on_batch(*item, **kwargs)
        return loss_sum / len(iterator)

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        metric: Optional = None,
        piecewise: bool = False,
        **kwargs,
    ) -> Tuple[float, torch.Tensor]:
        """Evaluates model output using a metric function.

        Parameters
        ----------
        x
            Input data.
        y
            Target data.
        metric
            Metric function. If None, uses fnet.metrics.corr_coef.
        piecewise
            Set to perform predictions piecewise.
        **kwargs
            Additional kwargs to be passed to predict() method.

        Returns
        -------
        float
            Evaluation as determined by metric function.
        torch.Tensor
            Model prediction.

        """
        if metric is None:
            metric = corr_coef
        if piecewise:
            y_hat = self.predict_piecewise(x, **kwargs)
        else:
            y_hat = self.predict(x, **kwargs)
        if y is None:
            return None, y_hat
        evaluation = metric(y, y_hat)
        return evaluation, y_hat

    def apply_on_single_zstack(
        self,
        input_img: Optional[np.ndarray] = None,
        filename: Optional[Union[Path, str]] = None,
        inputCh: Optional[int] = None,
        normalization: Optional[Callable] = None,
        already_normalized: bool = False,
        ResizeRatio: Optional[Sequence[float]] = None,
        cutoff: Optional[float] = None,
    ) -> np.ndarray:
        """Applies model to a single z-stack input.

        This assumes the loaded network architecture can receive 3d grayscale
        images as input.

        Parameters
        ----------
        input_img
            3d or 4d image with shape (Z, Y, X) or (C, Z, Y, X) respectively.
        filename
            Path to input image. Ignored if input_img is supplied.
        inputCh
            Selected channel if filename is a path to a 4d image.
        normalization
            Input image normalization function.
        already_normalized
            Set to skip input normalization.
        ResizeRatio
            Resizes each dimension of the the input image by the specified
            factor if specified.
        cutoff
            If specified, converts the output to a binary image with cutoff as
            threshold value.

        Returns
        -------
        np.ndarray
            Predicted image with shape (Z, Y, X). If cutoff is set, dtype will
            be numpy.uint8. Otherwise, dtype will be numpy.float.

        Raises
        ------
        ValueError
            If parameters are invalid.
        FileNotFoundError
            If specified file does not exist.
        IndexError
            If inputCh is invalid.

        """
        if input_img is None:
            if filename is None:
                raise ValueError("input_img or filename must be specified")
            input_img = tifffile.imread(str(filename))
        if inputCh is not None:
            if input_img.ndim != 4:
                raise ValueError("input_img must be 4d if inputCh specified")
            input_img = input_img[inputCh,]
        if input_img.ndim != 3:
            raise ValueError("input_img must be 3d")
        normalization = normalization or norm_around_center
        if not already_normalized:
            input_img = normalization(input_img)
        if ResizeRatio is not None:
            if len(ResizeRatio) != 3:
                raise ValueError("ResizeRatio must be length 3")
            input_img = zoom(input_img, zoom=ResizeRatio, mode="nearest")
        yhat = (
            self.predict_piecewise(input_img[np.newaxis,], tta=True)
            .squeeze(dim=0)
            .numpy()
        )
        if cutoff is not None:
            yhat = (yhat >= cutoff).astype(np.uint8) * 255
        return yhat
