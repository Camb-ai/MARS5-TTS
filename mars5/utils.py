import torch
import logging

def length_to_mask(length, offsets, max_len=None):
    """
    Convert tensor of lengths into a mask.

    Args:
        length (Tensor): a tensor of lengths, shape = (batch_size,)
        offsets (Tensor): a tensor of offsets, shape = (batch_size,)
        max_len (int, optional): maximum length to be considered

    Returns:
        mask (Tensor): a mask tensor, shape = (batch_size, max_len), 
                        True in masked positions, False otherwise.
    """
    # get the batch size
    batch_size = length.size(0)
    
    # if maximum length is not provided, then compute it from the 'length' tensor.
    if max_len is None:
        max_len = length.max().item()
    
    # Create a tensor of size `(batch_size, max_len)` filled with `True`.
    mask = torch.ones(size=(batch_size, max_len), dtype=torch.bool, device=length.device)
    
    # Create a tensor with consecutive numbers.
    range_tensor = torch.arange(max_len, device=length.device)
    
    # Expand the dim of 'length' tensor and 'offset' tensor to make it `(batch_size, max_len)`.
    # The added dimension will be used for broadcasting.
    length_exp = length.unsqueeze(-1)
    offsets_exp = offsets.unsqueeze(-1)
    
    # Create a boolean mask where `False` represents valid positions and `True` represents padding.
    mask = (range_tensor < offsets_exp) | (~(range_tensor < length_exp))

    return mask


def construct_padding_mask(input_tensor, pad_token):
    return (input_tensor == pad_token).cumsum(dim=1) > 0    


def nuke_weight_norm(module):
    """
    Recursively remove weight normalization from a module and its children.

    Args:
        module (torch.nn.Module): The module from which to remove weight normalization.
    """
    # Remove weight norm from current module if it exists
    try:
        torch.nn.utils.remove_weight_norm(module)
        logging.debug(f"Removed weight norm from {module.__class__.__name__}")
    except ValueError:
        # Ignore if the module does not have weight norm applied.
        pass

    # Recursively call the function on children modules
    for child in module.children():
        nuke_weight_norm(child)
