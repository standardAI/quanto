from functools import partial

import torch

from .core import QTensor


__all__ = ["get_qtensor_func", "register_qtensor_func"]


_QTENSOR_FUNC_TABLE = {}


def register_qtensor_func(funcs):
    """
    Used for registering a new __torch_dispatch__ function to QTensor

    The code to register a new function looks like:

    @register_qtensor_func(list_of_funcs)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(qfunc):
        for func in funcs:
            _QTENSOR_FUNC_TABLE[func] = partial(qfunc, func)

    return wrapper


def get_qtensor_func(func):
    return _QTENSOR_FUNC_TABLE.get(func, None)


def dequantize(*args):
    return [arg.dequantize() if isinstance(arg, QTensor) else arg for arg in args]


@register_qtensor_func([torch.nn.functional.log_softmax, torch.topk, torch.nn.functional.layer_norm])
def unary_unsupported_op(func, t, *args, **kwargs):
    return func(t.dequantize(), *args, **kwargs)


@register_qtensor_func([torch.nn.functional.cross_entropy, torch.nn.functional.cosine_similarity])
def plurary_unsupported_op(func, *args, **kwargs):
    return func(*dequantize(*args), **kwargs)


from torch_int._CUDA import linear_a8_w8_bfp32_ofp32


@register_qtensor_func([torch.nn.functional.linear])
def linear(func, input, other, bias=None):
    if isinstance(input, QTensor) and input.device.type == 'cuda' and input.itype == torch.int8 and isinstance(other, QTensor) and other.axis is None:
        input_shape = input.shape
        output_scale = input._scale * other._scale
        input = input.view(-1, input_shape[-1])
        if bias is None:
            bias = torch.zeros((input_shape[-1]), dtype=other.dtype)
        bias = bias.to(torch.float32)
        output = linear_a8_w8_bfp32_ofp32(input._data.contiguous(), other._data.contiguous(), bias, float(output_scale.item()), 1.0)
        return output.view(*input_shape[:-1], -1).to(input.dtype).contiguous()
    output = torch.matmul(input, other.t())
    if bias is not None:
        output = output + bias
    return output
