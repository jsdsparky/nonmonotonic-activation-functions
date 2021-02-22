# Modified from https://pytorch.org/docs/stable/notes/extending.html

import math
import torch
import torch.nn as nn

class BellCurveFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, bias):
        sig_x = torch.sigmoid(weight*input + bias)
        sig_x_2 = sig_x**2
        output = 4*(sig_x - sig_x_2)
        
        ctx.save_for_backward(input, weight, sig_x, sig_x_2)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, sig_x, sig_x_2 = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        sig_x_3 = sig_x_2 * sig_x
        poly_sig_x = 8*sig_x_3 - 12*sig_x_2 + 4*sig_x
        grad_out_poly = grad_output * poly_sig_x
        
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = weight * grad_out_poly
        if ctx.needs_input_grad[1]:
            grad_weight = input * grad_out_poly
        if ctx.needs_input_grad[2]:
            grad_bias = grad_out_poly

        return grad_input, grad_weight, grad_bias
