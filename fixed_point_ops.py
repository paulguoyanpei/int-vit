"""Fixed-point arithmetic operations and lookup tables.

Representation: real value v -> integer v_int = round(v * 2^Q)
All ops work on int64 tensors.
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Core quantize / dequantize
# ---------------------------------------------------------------------------

def quantize(x: torch.Tensor, Q: int) -> torch.Tensor:
    """Float -> fixed-point integer."""
    return (x * (1 << Q)).round().long()


def dequantize(x_int: torch.Tensor, Q: int) -> torch.Tensor:
    """Fixed-point integer -> float."""
    return x_int.float() / (1 << Q)


# ---------------------------------------------------------------------------
# Lookup Tables
# ---------------------------------------------------------------------------

class LookupTable:
    """Precomputed lookup table for a scalar function.

    Maps fixed-point integer inputs in [min_val_int, max_val_int] to
    fixed-point integer outputs. Out-of-range inputs are clamped.
    """

    def __init__(self, func, min_val: float, max_val: float, Q: int):
        """
        Args:
            func: Python callable, Tensor -> Tensor (or scalar float -> float)
            min_val: minimum real-valued input
            max_val: maximum real-valued input
            Q: fractional bits
        """
        self.Q = Q
        self.min_val = min_val
        self.max_val = max_val

        # Precompute bounds in fixed-point
        self.min_int = round(min_val * (1 << Q))
        self.max_int = round(max_val * (1 << Q))
        self.size = self.max_int - self.min_int + 1
        print("build table with size", self.size)

        # Precompute every Q-scale fixed-point integer in the closed range.
        input_ints = torch.arange(self.min_int, self.max_int + 1, dtype=torch.long)
        inputs = input_ints.float() / (1 << Q)
        try:
            outputs = func(inputs)
        except (TypeError, ValueError):
            outputs = torch.tensor([func(x.item()) for x in inputs])
        if not torch.is_tensor(outputs):
            outputs = torch.tensor(outputs)
        self.table = quantize(outputs, Q)  # int64 tensor of size [size]

        # For index computation: index = x_int_clamped - min_int
        self.offset = -self.min_int

    def lookup(self, x_int: torch.Tensor) -> torch.Tensor:
        """Look up fixed-point input in table, return fixed-point output."""
        x_clamped = x_int.clamp(self.min_int, self.max_int)
        idx = x_clamped - self.min_int
        return self.table[idx]


def build_gelu_lut(Q: int, min_val=-8.0, max_val=8.0) -> LookupTable:
    """GELU lookup table."""
    def gelu_fn(x):
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return LookupTable(gelu_fn, min_val, max_val, Q)


def build_exp_lut(Q: int, min_val=-16.0, max_val=0.0) -> LookupTable:
    """exp() lookup table. Input range is negative (for softmax stability)."""
    return LookupTable(torch.exp, min_val, max_val, Q)


def build_recip_lut(Q: int, min_val=0.01, max_val=512.0) -> LookupTable:
    """1/x lookup table for softmax normalization.

    Range covers sums up to 512 (well above 197 tokens).
    """
    return LookupTable(lambda x: torch.where(x > 0, 1.0 / x, torch.zeros_like(x)),
                       min_val, max_val, Q)


def build_rsqrt_lut(Q: int, min_val=0.001, max_val=256.0) -> LookupTable:
    """1/sqrt(x) lookup table for LayerNorm."""
    return LookupTable(lambda x: torch.where(x > 0, torch.rsqrt(x), torch.zeros_like(x)),
                       min_val, max_val, Q)


# ---------------------------------------------------------------------------
# Fixed-point layers
# ---------------------------------------------------------------------------

class FixedPointLinear:
    """Fixed-point matrix multiply: out = (x_int @ W_int + b_int * 2^Q) >> Q"""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, Q: int):
        self.Q = Q
        self.W_int = quantize(weight, Q)  # [out, in]
        self.b_int = quantize(bias, Q) if bias is not None else None

    def __call__(self, x_int: torch.Tensor) -> torch.Tensor:
        # x_int: [..., in_features], W_int: [out, in] -> matmul with W^T
        out = (x_int @ self.W_int.t()) >> self.Q
        if self.b_int is not None:
            out = out + self.b_int
        return out


class FixedPointLayerNorm:
    """Fixed-point LayerNorm using rsqrt LUT.

    y = (x - mean) * rsqrt(var + eps) * gamma + beta
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, eps: float,
                 Q: int, rsqrt_lut: LookupTable):
        self.Q = Q
        self.eps = eps
        self.gamma_int = quantize(weight, Q)
        self.beta_int = quantize(bias, Q)
        self.rsqrt_lut = rsqrt_lut

    def __call__(self, x_int: torch.Tensor) -> torch.Tensor:
        # Mean: sum / n (integer division)
        n = x_int.shape[-1]
        mean_int = x_int.sum(dim=-1, keepdim=True) // n

        # Centered
        centered = x_int - mean_int

        # Variance: sum(centered^2) / n, but centered^2 is in 2^(2Q) scale
        # so var_int is in 2^(2Q) scale after division by n
        var_int_2q = (centered * centered).sum(dim=-1, keepdim=True) // n

        # Convert variance to real scale for rsqrt LUT:
        # var_int_2q is in 2^(2Q) scale, we need var in 2^Q scale for the LUT
        # var_real = var_int_2q / 2^(2Q), we want quantize(var_real, Q) = var_int_2q >> Q
        var_int = var_int_2q >> self.Q

        # Add eps in fixed-point
        eps_int = round(self.eps * (1 << self.Q))
        var_int = var_int + eps_int

        # rsqrt via LUT: need to dequantize var, look up rsqrt, get result in Q scale
        # The LUT expects input in Q-scale fixed point representing real values
        rsqrt_int = self.rsqrt_lut.lookup(var_int)

        # Normalize: (centered * rsqrt) >> Q
        normed = (centered * rsqrt_int) >> self.Q

        # Scale and shift: ((normed * gamma) >> Q) + beta
        out = ((normed * self.gamma_int) >> self.Q) + self.beta_int
        return out


class FixedPointGELU:
    """GELU via lookup table."""

    def __init__(self, Q: int, gelu_lut: LookupTable):
        self.gelu_lut = gelu_lut

    def __call__(self, x_int: torch.Tensor) -> torch.Tensor:
        return self.gelu_lut.lookup(x_int)


class FixedPointSoftmax:
    """Softmax via exp LUT + reciprocal LUT.

    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """

    def __init__(self, Q: int, exp_lut: LookupTable, recip_lut: LookupTable):
        self.Q = Q
        self.exp_lut = exp_lut
        self.recip_lut = recip_lut

    def __call__(self, x_int: torch.Tensor) -> torch.Tensor:
        # Subtract max for numerical stability (in integer domain)
        x_max = x_int.max(dim=-1, keepdim=True).values
        x_shifted = x_int - x_max  # all <= 0

        # exp via LUT (input is negative)
        exp_int = self.exp_lut.lookup(x_shifted)

        # Sum of exps
        exp_sum = exp_int.sum(dim=-1, keepdim=True)

        # Reciprocal of sum via LUT
        # exp_sum is in Q-scale. The recip LUT expects real-valued input in Q-scale.
        recip_sum = self.recip_lut.lookup(exp_sum)

        # softmax = exp * recip_sum >> Q
        out = (exp_int * recip_sum) >> self.Q
        return out
