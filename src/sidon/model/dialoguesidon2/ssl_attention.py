"""Memory-efficient relative-key self-attention for w2v-BERT 2.0.

w2v-BERT 2.0 adds a learned bias q_i . E[clip(j - i)] to every attention score, where E holds one embedding per
clipped relative distance (73 distances: 64 to the left, 8 to the right). The reference implementation builds a
[T, T, D] tensor of distance embeddings. ``RelativeKeyAttention`` computes the same scores by first projecting the
queries onto the 73 embeddings ([B, H, T, 73]) and then gathering the bias per score, and runs the attention with
``scaled_dot_product_attention``. It reuses the original module's parameters, so state-dict keys are unchanged.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def relative_flex_attention(query: Tensor, key: Tensor, value: Tensor, buckets: Tensor, left: int) -> Tensor:
    """Attention with the relative-key bias added inside a FlexAttention score function.

    buckets [B, H, T, K] holds q_i . E[d] for each clipped distance d; the score of (i, j) gets
    buckets[b, h, i, clip(j - i) + left] / sqrt(head_dim), so no [B, H, T, T] bias is materialized. Used only under
    torch.compile and without gradients (the frozen encoders); eager mode falls back to a dense reference.
    """
    from torch.nn.attention.flex_attention import flex_attention

    right = buckets.size(-1) - 1 - left
    scale = math.sqrt(query.size(-1))

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + buckets[b, h, q_idx, torch.clamp(kv_idx - q_idx, -left, right) + left] / scale

    return flex_attention(query, key, value, score_mod=score_mod)


class RelativeKeyAttention(nn.Module):
    """Drop-in replacement of ``Wav2Vec2BertSelfAttention`` with relative_key position embeddings."""

    def __init__(self, original: nn.Module):
        super().__init__()
        if original.position_embeddings_type != "relative_key":
            raise ValueError("Only relative_key attention supports this optimization")
        for name in ("head_size", "num_heads", "position_embeddings_type",
                     "left_max_position_embeddings", "right_max_position_embeddings",
                     "linear_q", "linear_k", "linear_v", "linear_out", "dropout", "distance_embedding"):
            setattr(self, name, getattr(original, name))
        self.training = original.training
        # Execution option set by compile_training_blocks (performance.flex_ssl_attention); not a weight.
        self.flex = False

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None,
                relative_position_embeddings: Tensor | None = None,
                output_attentions: bool = False) -> tuple[Tensor, Tensor | None]:
        batch, frames, _ = hidden_states.shape

        def split(value: Tensor) -> Tensor:
            return value.view(batch, frames, self.num_heads, self.head_size).transpose(1, 2)

        query = split(self.linear_q(hidden_states))
        key = split(self.linear_k(hidden_states))
        value = split(self.linear_v(hidden_states))
        scale = math.sqrt(self.head_size)
        positions = torch.arange(frames, device=hidden_states.device)
        distance = positions[None, :] - positions[:, None]
        distance.clamp_(-self.left_max_position_embeddings, self.right_max_position_embeddings)
        distance.add_(self.left_max_position_embeddings)
        # Queries projected onto every distance embedding: [B, H, T, D] @ [D, K] -> [B, H, T, K].
        buckets = torch.matmul(query, self.distance_embedding.weight.to(query.dtype).T)
        needs_grad = buckets.requires_grad and torch.is_grad_enabled()
        if (self.flex and not needs_grad and attention_mask is None and not output_attentions
                and torch.compiler.is_compiling() and (self.dropout.p == 0 or not self.training)):
            output = relative_flex_attention(query, key, value, buckets, self.left_max_position_embeddings)
            return self.linear_out(output.transpose(1, 2).reshape(batch, frames, -1)), None
        if needs_grad:
            # A plain gather would send the gradients of every clipped score into the two end buckets one by one.
            # The interior distances are scattered instead, and the two clipped regions are filled by broadcasting.
            if buckets.size(-1) == 1:
                relative_scores = buckets.expand(batch, self.num_heads, frames, frames)
            else:
                offsets = torch.arange(1 - self.left_max_position_embeddings,
                                       self.right_max_position_embeddings, device=query.device)
                indices = positions[:, None] + offsets[None, :]  # [T, K-2]
                valid = (indices >= 0) & (indices < frames)
                interior = buckets[..., 1:-1].masked_fill(~valid, 0)
                relative_scores = query.new_zeros(batch, self.num_heads, frames, frames).scatter_add(
                    -1, indices.clamp(0, frames - 1)[None, None].expand(batch, self.num_heads, -1, -1),
                    interior)
                relative_scores = torch.where(distance[None, None] == 0, buckets[..., :1], relative_scores)
                relative_scores = torch.where(distance[None, None] == buckets.size(-1) - 1,
                                              buckets[..., -1:], relative_scores)
        else:
            relative_scores = buckets.gather(-1, distance[None, None].expand(batch, self.num_heads, -1, -1))
        bias = relative_scores / scale
        if attention_mask is not None:
            bias = bias + attention_mask.to(query.dtype)
        probabilities = None
        if output_attentions:
            # scaled_dot_product_attention does not return the weights; keep the transformers interface.
            scores = torch.matmul(query, key.transpose(-2, -1)) / scale + bias
            probabilities = self.dropout(torch.softmax(scores, dim=-1))
            output = torch.matmul(probabilities, value)
        else:
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=bias,
                dropout_p=self.dropout.p if self.training else 0.0)
        output = output.transpose(1, 2).reshape(batch, frames, -1)
        return self.linear_out(output), probabilities


def optimize_relative_key_attention(model: nn.Module) -> int:
    """Replace every relative_key ``Wav2Vec2BertSelfAttention`` in `model` in place; returns the count."""
    from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertSelfAttention

    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, Wav2Vec2BertSelfAttention) and child.position_embeddings_type == "relative_key":
            setattr(model, name, RelativeKeyAttention(child))
            count += 1
        else:
            count += optimize_relative_key_attention(child)
    return count
