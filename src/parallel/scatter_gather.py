import torch
import torch.nn.parallel.scatter_gather as torch_
from src.sparse_torch import CSRMatrix3d, CSCMatrix3d, concatenate
import torch_geometric as pyg
import torch.nn.functional as F


def scatter(inputs, target_gpus, dim=0):
    """
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return torch_.Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))

        # modified here
        if isinstance(obj, CSRMatrix3d) or isinstance(obj, CSCMatrix3d):
            return scatter_sparse_matrix(target_gpus, obj)
        
        # * for DataBatch
        if isinstance(obj, pyg.data.Batch):
            return scatter_pyg_batch(target_gpus, obj)

        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

# * for DataBatch
def scatter_pyg_batch(target_gpus, obj):
    def get_device(i):
        return torch.device('cuda:{}'.format(i)) if i != -1 else torch.device('cpu')
    step = obj.num_graphs // len(target_gpus)
    data_list = [obj.to_data_list()[i:i+step] for i in range(0, obj.num_graphs, step)]
    return tuple([[data.to(get_device(i // step)) for data in data_list[i]] for i in range(len(data_list))])

# * for DataBatch
def gather_pyg_batch(outputs, target_device):
    data_list = []
    for out in outputs:
        data_list.extend(out.to_data_list())
    data_list = [data.to(target_device) for data in data_list]
    return pyg.data.Batch.from_data_list(data_list).to(target_device)

def scatter_sparse_matrix(target_gpus, obj):
    """Scatter for customized sparse matrix"""
    def get_device(i):
        return torch.device('cuda:{}'.format(i)) if i != -1 else torch.device('cpu')
    step = len(obj) // len(target_gpus)
    return tuple([obj[i:i+step].to(get_device(i // step)) for i in range(0, len(obj), step)])


def gather(outputs, target_device, dim=0):
    """
    Gathers tensors from different GPUs on a specified device (-1 means the CPU).
    """
    
    def gather_map(outputs):
        out = outputs[0]
        try:
            if out.shape[-1] > outputs[1].shape[-1]:
                outputs[1] = F.pad(outputs[1], pad=(0, out.shape[-1]-outputs[1].shape[-1], 0, 0, 0, 0), mode='constant', value=0)
            elif out.shape[-1] < outputs[1].shape[-1]:
                out = F.pad(out, pad=(0, outputs[1].shape[-1]-out.shape[-1], 0, 0, 0, 0), mode='constant', value=0)
                outputs[0] = out
            
            if out.shape[1] > outputs[1].shape[1]:
                outputs[1] = F.pad(outputs[1], pad=(0, 0, 0, out.shape[1]-outputs[1].shape[1], 0, 0), mode='constant', value=0)
            elif out.shape[1] < outputs[1].shape[1]:
                out = F.pad(out, pad=(0, 0, 0, outputs[1].shape[1]-out.shape[1], 0, 0), mode='constant', value=0)
                outputs[0] = out
        except:
            pass
        
        if isinstance(out, torch.Tensor):
            return torch_.Gather.apply(target_device, dim, *outputs)

        # modified here
        if isinstance(out, CSRMatrix3d) or isinstance(out, CSCMatrix3d):
            return concatenate(*outputs, device=target_device)
        
        # * for DataBatch
        if isinstance(out, pyg.data.Batch):
            return gather_pyg_batch(outputs, target_device)

        if out is None:
            return None

        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        if isinstance(out, int):
            assert all([out == _ for _ in outputs])
            return out

        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None
