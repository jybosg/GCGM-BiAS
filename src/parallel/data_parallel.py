import torch.nn as nn
from .scatter_gather import scatter_kwargs, gather
from torch.nn.parallel import DistributedDataParallel as DDP


class DataParallel(nn.DataParallel):
# class DataParallel(nn.parallel.DistributedDataParallel):
    """
    DataParallel wrapper with customized scatter/gather functions
    """
    def __init__(self, *args, **kwargs):
        super(DataParallel, self).__init__(*args, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)
