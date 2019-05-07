import logging

from tqdm import tqdm
import numpy as np


logger = logging.getLogger(__name__)


class BufferedPatchDataset:
    """Provides patches from items of a dataset."""

    def __init__(
            self,
            dataset,
            patch_size=(32, 64, 64),
            buffer_size=1,
            buffer_switch_frequency=-1,
            shuffle_images=True,
    ):
        self.counter = 0
        self.dataset = dataset
        self.buffer_size = min(len(self.dataset), buffer_size)
        self.buffer_switch_frequency = buffer_switch_frequency
        self.patch_size = patch_size
        self.buffer = list()
        self.shuffle_images = shuffle_images
        shuffed_data_order = np.arange(0, len(dataset))
        if self.shuffle_images:
            np.random.shuffle(shuffed_data_order)
        pbar = tqdm(range(0, self.buffer_size))
        pbar.set_description('Buffering images')
        self.buffer_history = list()
        for i in pbar:
            datum_index = shuffed_data_order[i]
            datum = dataset[datum_index]
            self.buffer_history.append(datum_index)
            self.buffer.append(datum)
        self.remaining_to_be_in_buffer = shuffed_data_order[i + 1:]

    def __iter__(self):
        return self

    def __next__(self):
        patch = self.get_random_patch()
        self.counter += 1
        if (self.buffer_switch_frequency > 0) and (
            self.counter % self.buffer_switch_frequency == 0
        ):
            self.insert_new_element_into_buffer()
        return patch

    def __getitem__(self, index):
        self.counter += 1
        if (self.buffer_switch_frequency > 0) and (
            self.counter % self.buffer_switch_frequency == 0
        ):
            self.insert_new_element_into_buffer()
        return self.get_random_patch()

    def insert_new_element_into_buffer(self):
        # sample with replacement
        self.buffer.pop(0)
        if self.shuffle_images:

            if len(self.remaining_to_be_in_buffer) == 0:
                self.remaining_to_be_in_buffer = np.arange(
                    0, len(self.dataset)
                )
                np.random.shuffle(self.remaining_to_be_in_buffer)

            new_datum_index = self.remaining_to_be_in_buffer[0]
            self.remaining_to_be_in_buffer = self.remaining_to_be_in_buffer[1:]

        else:
            new_datum_index = self.buffer_history[-1] + 1
            if new_datum_index == len(self.dataset):
                new_datum_index = 0

        self.buffer_history.append(new_datum_index)
        self.buffer.append(self.dataset[new_datum_index])
        logger.info(f"Added item {new_datum_index} into buffer")

    def get_random_patch(self):
        """Samples random patch from an item in the buffer.

        Let nd be the number of dimensions of the patch. If the item has more
        dimensions than the patch size, then sampling will be from the last nd
        dimensions of the item.

        """
        nd = len(self.patch_size)
        buffer_index = np.random.randint(len(self.buffer))
        datum = self.buffer[buffer_index]
        shape_spatial = datum[0].shape[-nd:]
        if (
                nd > len(shape_spatial)
                or any(
                    self.patch_size[d] > shape_spatial[d] for d in range(nd)
                )
        ):
            raise ValueError(
                f'Incompatible patch size {self.patch_size} and dataset '
                f'item shape (index: {buffer_index}, shape: {shape_spatial})'
            )
        patch = []
        slices = None
        for idx_p, part in enumerate(datum):
            if part.shape[-nd:] != shape_spatial:
                raise ValueError(
                    f'Datum component {idx_p} spatial shape '
                    f'{part.shape[-nd:]} incompatible with first component '
                    f'spatial shape {shape_spatial}'
                )
            if slices is None:
                starts = np.array(
                    [
                        np.random.randint(0, d - p + 1)
                        for d, p in zip(shape_spatial, self.patch_size)
                    ]
                )
                ends = starts + np.array(self.patch_size)
                slices = tuple(slice(s, e) for s, e in zip(starts, ends))
            # Pad slices with "slice(None)" if there are non-spatial dimensions
            slices_pad = (slice(None),)*(len(part.shape) - len(shape_spatial))
            patch.append(part[slices_pad + slices])
        return patch

    def get_buffer_history(self):
        return self.buffer_history
