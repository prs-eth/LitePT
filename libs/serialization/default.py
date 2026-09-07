import torch
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_

try:
    import serialization_cuda  # run `python setup.py install`
    # print("Has CUDA-accelerated serialization.")
    HAS_CUDA_EXT = True
except ImportError:
    # print("No CUDA-accelerated serialization.")
    HAS_CUDA_EXT = False

# import numpy as np
# from colorhash import ColorHash

# def int_to_plotly_rgb(x):
#     """Convert 1D torch.Tensor of int into plotly-friendly RGB format.
#     This operation is deterministic on the int values.
#     """
#     assert isinstance(x, torch.Tensor)
#     assert x.dim() == 1
#     assert not x.is_floating_point()
#     x = x.cpu().long().numpy()
#     palette = np.array([ColorHash(i).rgb for i in range(x.max() + 1)])
#     return palette[x]

@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
        # code = torch.argsort(code)
        # rgb = int_to_plotly_rgb(code)
        # print('a')
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
        # code = torch.argsort(code)
        # rgb = int_to_plotly_rgb(code)
        # print('a')
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    if HAS_CUDA_EXT:
        code_cuda = serialization_cuda.morton_encode(grid_coord).long()
        # basic differential testing of the CUDA implementation
        # x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
        # # we block the support to batch, maintain batched code in Point class
        # code_torch = z_order_encode_(x, y, z, b=None, depth=depth)
        # print("morton equal", torch.all(code_cuda == code_torch))
        return code_cuda
    else:
        x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
        # we block the support to batch, maintain batched code in Point class
        code = z_order_encode_(x, y, z, b=None, depth=depth)
        return code


def z_order_decode(code: torch.Tensor, depth):
    x, y, z = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord



def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    if HAS_CUDA_EXT:
        code_cuda = serialization_cuda.hilbert_encode(grid_coord.to(torch.uint32).contiguous(), depth).long()
        # basic differential testing of the CUDA implementation
        # code_torch = hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)
        # print("hilbert equal", torch.all(code_cuda == code_torch))
        return code_cuda
    else:
        return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)
