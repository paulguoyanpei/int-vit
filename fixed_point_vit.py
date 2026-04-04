"""Fixed-point ViT: wraps a pretrained DeiT-Tiny for integer-only inference.

All computation happens on int64 tensors with Q fractional bits.
Non-linear functions (GELU, Softmax exp/recip, LayerNorm rsqrt) use LUTs.
"""

import torch
import math
from fixed_point_ops import (
    quantize, dequantize,
    FixedPointLinear, FixedPointLayerNorm, FixedPointGELU, FixedPointSoftmax,
    build_gelu_lut, build_exp_lut, build_recip_lut, build_rsqrt_lut,
)


class FixedPointAttention:
    """Multi-head self-attention in fixed-point."""

    def __init__(self, attn_module, Q, exp_lut, recip_lut):
        self.Q = Q
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        # scale = 1/sqrt(head_dim), quantized
        self.scale_int = round((1.0 / math.sqrt(self.head_dim)) * (1 << Q))

        self.qkv = FixedPointLinear(attn_module.qkv.weight.data,
                                    attn_module.qkv.bias.data, Q)
        self.proj = FixedPointLinear(attn_module.proj.weight.data,
                                     attn_module.proj.bias.data, Q)
        self.softmax = FixedPointSoftmax(Q, exp_lut, recip_lut)

    def __call__(self, x_int):
        B, N, C = x_int.shape

        # QKV projection
        qkv = self.qkv(x_int)  # [B, N, 3*C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores: (Q @ K^T) * scale >> Q
        attn = (q @ k.transpose(-2, -1))  # [B, H, N, N] in 2^(2Q) scale
        attn = (attn * self.scale_int) >> (2 * self.Q)  # back to Q scale

        # Softmax
        attn = self.softmax(attn)

        # Attention output: attn @ V >> Q
        out = (attn @ v) >> self.Q  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, C)

        # Output projection
        out = self.proj(out)
        return out


class FixedPointMLP:
    """MLP block: Linear -> GELU -> Linear, all in fixed-point."""

    def __init__(self, mlp_module, Q, gelu_lut):
        self.fc1 = FixedPointLinear(mlp_module.fc1.weight.data,
                                    mlp_module.fc1.bias.data, Q)
        self.fc2 = FixedPointLinear(mlp_module.fc2.weight.data,
                                    mlp_module.fc2.bias.data, Q)
        self.gelu = FixedPointGELU(Q, gelu_lut)

    def __call__(self, x_int):
        x_int = self.fc1(x_int)
        x_int = self.gelu(x_int)
        x_int = self.fc2(x_int)
        return x_int


class FixedPointBlock:
    """Transformer block: LN -> Attn -> residual -> LN -> MLP -> residual."""

    def __init__(self, block_module, Q, gelu_lut, exp_lut, recip_lut, rsqrt_lut):
        self.norm1 = FixedPointLayerNorm(
            block_module.norm1.weight.data, block_module.norm1.bias.data,
            block_module.norm1.eps, Q, rsqrt_lut
        )
        self.attn = FixedPointAttention(block_module.attn, Q, exp_lut, recip_lut)
        self.norm2 = FixedPointLayerNorm(
            block_module.norm2.weight.data, block_module.norm2.bias.data,
            block_module.norm2.eps, Q, rsqrt_lut
        )
        self.mlp = FixedPointMLP(block_module.mlp, Q, gelu_lut)

    def __call__(self, x_int):
        # Self-attention with residual
        x_int = x_int + self.attn(self.norm1(x_int))
        # MLP with residual
        x_int = x_int + self.mlp(self.norm2(x_int))
        return x_int


class FixedPointViT:
    """Full fixed-point Vision Transformer for inference.

    Converts a pretrained timm VisionTransformer to fixed-point.
    Input: float images (will be quantized internally).
    Output: float logits (dequantized at the end).
    """

    def __init__(self, fp_model, Q):
        self.Q = Q

        # Build LUTs (shared across all layers)
        print(f"Building LUTs with Q={Q}...")
        self.gelu_lut = build_gelu_lut(Q)
        self.exp_lut = build_exp_lut(Q)
        self.recip_lut = build_recip_lut(Q)
        self.rsqrt_lut = build_rsqrt_lut(Q)

        # Quantize patch embedding
        pe = fp_model.patch_embed
        self.patch_proj = FixedPointLinear(
            pe.proj.weight.data.flatten(2).transpose(1, 2).reshape(-1, pe.proj.weight.shape[1] * 16 * 16),
            pe.proj.bias.data, Q
        )
        # Actually, Conv2d patch embed is easier to handle via direct quantization
        # Let's just store quantized conv weight and do it manually
        self.patch_weight_int = quantize(pe.proj.weight.data, Q)  # [192, 3, 16, 16]
        self.patch_bias_int = quantize(pe.proj.bias.data, Q)  # [192]

        # Quantize positional embedding and cls token
        self.cls_token_int = quantize(fp_model.cls_token.data, Q)  # [1, 1, 192]
        self.pos_embed_int = quantize(fp_model.pos_embed.data, Q)  # [1, 197, 192]

        # Build fixed-point blocks
        self.blocks = []
        for block in fp_model.blocks:
            self.blocks.append(FixedPointBlock(
                block, Q,
                self.gelu_lut, self.exp_lut, self.recip_lut, self.rsqrt_lut
            ))

        # Final LayerNorm
        self.norm = FixedPointLayerNorm(
            fp_model.norm.weight.data, fp_model.norm.bias.data,
            fp_model.norm.eps, Q, self.rsqrt_lut
        )

        # Classification head
        self.head = FixedPointLinear(fp_model.head.weight.data,
                                     fp_model.head.bias.data, Q)

    def patch_embed(self, x_int):
        """Fixed-point patch embedding via conv-like operation.

        x_int: [B, 3, 224, 224] in Q-scale int64
        Returns: [B, 196, 192] in Q-scale int64
        """
        B = x_int.shape[0]
        # Unfold into patches: [B, 3*16*16, 196]
        patches = x_int.unfold(2, 16, 16).unfold(3, 16, 16)  # [B, 3, 14, 14, 16, 16]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, 196, -1)  # [B, 196, 768]

        # Linear projection: patches @ weight^T >> Q + bias
        W = self.patch_weight_int.reshape(192, -1)  # [192, 768]
        out = (patches @ W.t()) >> self.Q
        out = out + self.patch_bias_int
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor [B, 3, 224, 224] (normalized images)
        Returns:
            logits: float tensor [B, num_classes]
        """
        # Quantize input
        x_int = quantize(x, self.Q)

        # Patch embedding
        x_int = self.patch_embed(x_int)  # [B, 196, 192]

        # Prepend cls token
        B = x_int.shape[0]
        cls_tokens = self.cls_token_int.expand(B, -1, -1)
        x_int = torch.cat([cls_tokens, x_int], dim=1)  # [B, 197, 192]

        # Add positional embedding
        x_int = x_int + self.pos_embed_int

        # Transformer blocks
        for block in self.blocks:
            x_int = block(x_int)

        # Final norm
        x_int = self.norm(x_int)

        # CLS token -> head
        cls_out = x_int[:, 0]  # [B, 192]
        logits_int = self.head(cls_out)  # [B, num_classes]

        # Dequantize output
        return dequantize(logits_int, self.Q)
