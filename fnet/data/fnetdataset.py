import torch.utils.data

class FnetDataset(torch.utils.data.Dataset):
    """Abstract class for fnet datasets."""

    def name(self, index) -> str:
        """Returns str to identify dataset element specified by index."""
        raise NotImplementedError

