
import torch

try:
    import pointrope as _kernels # run `python setup.py install`
except ModuleNotFoundError:
    from . import pointrope as _kernels # run `python setup.py build_ext --inplace`


class PointROPE_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        # tokens = tokens.clone() # uncomment this if inplace doesn't work
        _kernels.pointrope( tokens, positions, base, F0 )
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        _kernels.pointrope( grad_res, positions, base, -F0 )
        ctx.mark_dirty(grad_res)
        return grad_res, None, None, None


class PointROPE(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0

    def forward(self, tokens, positions): 
        PointROPE_func.apply(tokens.transpose(1,2), positions, self.base, self.F0)
        return tokens