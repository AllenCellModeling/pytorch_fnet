import torch.utils.data

class FnetDataset(torch.utils.data.Dataset):
    """Abstract class for fnet datasets."""

    def name(self, index) -> str:
        """Returns str to identify dataset element specified by index."""
        raise NotImplementedError

    def apply_transforms(self, pytorch_tensor):
        
        for t in self.transforms:
            pytorch_tensor = t(pytorch_tensor)
            
            
        return pytorch_tensor