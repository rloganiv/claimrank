"""
Masked activation functions
"""
import torch


def masked_softmax(vector, mask):
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
    return result

