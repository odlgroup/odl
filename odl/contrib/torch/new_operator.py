from __future__ import division

import warnings

from torch.autograd import Function
import itertools

import numpy as np
import torch
from packaging.version import parse as parse_version

from odl import Operator

if parse_version(torch.__version__) < parse_version('0.4'):
    warnings.warn("This interface is designed to work with Pytorch >= 0.4",
                  RuntimeWarning, stacklevel=2)

__all__ = ('OperatorFunction', 'OperatorModule')

class OperatorFunction(Function):
    @staticmethod
    def forward(
        ctx, 
        input_tensor:torch.Tensor,
        operator:Operator,   
        ):
        assert len(input_tensor.size()) == 5
        extra_dims = input_tensor.size()[:2]
        if input_tensor.requires_grad:
            ctx.operator     = operator
            ctx.extra_dims   = extra_dims
        
        output = input_tensor.new_empty(extra_dims + operator.range.shape, dtype=torch.float32) # type:ignore

        for subspace in itertools.product(*[range(dim_size) for dim_size in extra_dims]):
            output[subspace] = operator(input_tensor[subspace]).data # type:ignore
        return output

    @staticmethod
    def backward(ctx, grad_output):
        operator = ctx.operator
        grad_input = grad_output.new_empty(ctx.extra_dims + operator.domain.shape, dtype=torch.float32)  # type:ignore
        
        for subspace in itertools.product(*[range(dim_size) for dim_size in ctx.extra_dims]):
            grad_input[subspace] = operator.adjoint(grad_output[subspace]).data
        
        return grad_input, None

class OperatorModule(torch.nn.Module):

    def __init__(self, operator:Operator):
        """Initialize a new instance."""
        super(OperatorModule, self).__init__()
        self.operator = operator

    def forward(self, input_tensor:torch.Tensor):
        return OperatorFunction.apply(
            input_tensor,
            self.operator
            )

    def __repr__(self):
        """Return ``repr(self)``."""
        op_name = self.operator.__class__.__name__
        op_in_shape = self.operator.domain.shape #type:ignore
        if len(op_in_shape) == 1:
            op_in_shape = op_in_shape[0]
        op_out_shape = self.operator.range.shape #type:ignore
        if len(op_out_shape) == 1:
            op_out_shape = op_out_shape[0]

        return '{}({}) ({} -> {})'.format(
            self.__class__.__name__, op_name, op_in_shape, op_out_shape
        )