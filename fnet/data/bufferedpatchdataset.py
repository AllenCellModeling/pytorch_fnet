from collections import deque
from typing import List, Sequence, Union
import collections.abc
import logging

from tqdm import tqdm
import numpy as np
import torch


logger = logging.getLogger(__name__)
ArrayLike = Union[np.ndarray, torch.Tensor]


class BufferedPatchDataset:
    """Provides patches from items of a dataset.

    Parameters
    ----------
    dataset
        Dataset object.
    patch_shape
        Shape of patch to be extracted from dataset items.
    buffer_size
        Size of buffer.
    buffer_switch_interval
        Number of patches provided between buffer item exchanges. Set to -1 to
        disable exchanges.
    shuffle_images
        Set to randomize order of dataset item insertion into buffer.

    """

    def __init__(
        self,
        dataset: collections.abc.Sequence,
        patch_shape: Sequence[int] = (32, 64, 64),
        buffer_size: int = 1,
        buffer_switch_interval: int = -1,
        shuffle_images: bool = True,
    ):
        self.dataset = dataset
        self.patch_shape = patch_shape
        self.buffer_size = min(len(self.dataset), buffer_size)
        self.buffer_switch_interval = buffer_switch_interval
        self.shuffle_images = shuffle_images

        self.counter = 0
        self.epochs = -1  # incremented to 0 when buffer initially filled
        self.buffer = deque()
        self.remaining_to_be_in_buffer = deque()
        self.buffer_history = []
        for _ in tqdm(range(self.buffer_size), desc="Buffering images"):
            self.insert_new_element_into_buffer()

    def __iter__(self):
        return self

    def __next__(self):
        patch = self.get_random_patch()
        self.counter += 1
        if (self.buffer_switch_interval > 0) and (
            self.counter % self.buffer_switch_interval == 0
        ):
            self.insert_new_element_into_buffer()
        return patch

    def _check_last_datum(self) -> None:
        """Checks last dataset item added to buffer."""
        nd = len(self.patch_shape)
        idx_buf = self.buffer_history[-1]
        shape_spatial = None
        for idx_c, component in enumerate(self.buffer[-1]):
            if shape_spatial is None:
                shape_spatial = component.shape[-nd:]
            elif component.shape[-nd:] != shape_spatial:
                raise ValueError(
                    f"Dataset item {idx_buf}, component {idx_c} shape "
                    f"{component.shape} incompatible with first component "
                    f"shape {self.buffer[-1][0].shape}"
                )
            if nd > len(component.shape) or any(
                self.patch_shape[d] > shape_spatial[d] for d in range(nd)
            ):
                raise ValueError(
                    f"Dataset item {idx_buf}, component {idx_c} shape "
                    f"{component.shape} incompatible with patch_shape "
                    f"{self.patch_shape}"
                )

    def insert_new_element_into_buffer(self) -> None:
        """Inserts new dataset item into buffer.

        Returns
        -------
        None

        """
        if len(self.remaining_to_be_in_buffer) == 0:
            self.epochs += 1
            self.remaining_to_be_in_buffer = deque(range(len(self.dataset)))
            if self.shuffle_images:
                np.random.shuffle(self.remaining_to_be_in_buffer)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.popleft()
        new_datum_index = self.remaining_to_be_in_buffer.popleft()
        self.buffer_history.append(new_datum_index)
        self.buffer.append(self.dataset[new_datum_index])
        logger.info(f"Added item {new_datum_index} into buffer")
        self._check_last_datum()

    def get_random_patch(self) -> List[ArrayLike]:
        """Samples random patch from an item in the buffer.

        Let nd be the number of dimensions of the patch. If the item has more
        dimensions than the patch, then sampling will be from the last nd
        dimensions of the item.

        Returns
        -------
        List[ArrayLike]
            Random patch sampled from a dataset item.

        """
        nd = len(self.patch_shape)
        buffer_index = np.random.randint(len(self.buffer))
        datum = self.buffer[buffer_index]
        shape_spatial = datum[0].shape[-nd:]
        patch = []
        slices = None
        for part in datum:
            if slices is None:
                starts = np.array(
                    [
                        np.random.randint(0, d - p + 1)
                        for d, p in zip(shape_spatial, self.patch_shape)
                    ]
                )
                ends = starts + np.array(self.patch_shape)
                slices = tuple(slice(s, e) for s, e in zip(starts, ends))
            # Pad slices with "slice(None)" if there are non-spatial dimensions
            slices_pad = (slice(None),) * (len(part.shape) - len(shape_spatial))
            patch.append(part[slices_pad + slices])
        return patch

    def get_batch(self, batch_size: int) -> Sequence[torch.Tensor]:
        """Returns a batch of patches.

        Parameters
        ----------
        batch_size
            Number of patches in batch.

        Returns
        -------
        Sequence[torch.Tensor]
            Batch of patches.

        """
        return tuple(
            torch.tensor(np.stack(part))
            for part in zip(*[next(self) for _ in range(batch_size)])
        )

    def get_buffer_history(self) -> List[int]:
        """Returns a list of indices of dataset elements inserted into the
        buffer.

        Returns
        -------
        List[int]
            Indices of dataset elements.

        """
        return self.buffer_history
