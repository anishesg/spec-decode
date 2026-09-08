"""
spec_decode: fused GPU-resident speculative decoding verification.
"""

from __future__ import annotations
import torch
from typing import Dict

try:
    from spec_decode import _C as _ext
    _HAS_CUDA_EXT = True
except ImportError:
    _HAS_CUDA_EXT = False


def verify_speculative(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Run speculative decoding verification for a single instance.

    Args:
        target_logits: float16 or float32 tensor of shape [k+1, vocab_size].
            Row i contains the target model's logits at draft position i.
            Row k is used for bonus token sampling when all drafts are accepted.
        draft_logits:  float16 or float32 tensor of shape [k, vocab_size].
        draft_tokens:  int64 tensor of shape [k]. Token IDs proposed by draft model.
        temperature:   softmax temperature applied to both logit sets before verification.
                       1.0 = no rescaling.

    Returns:
        dict with:
            num_accepted (int):               number of accepted draft tokens (0..k)
            tokens (Tensor, shape [k+1]):     accepted tokens + 1 correction/bonus token
            log_probs (Tensor, shape [k+1]):  log probability under target for each token
    """
    if not _HAS_CUDA_EXT:
        raise RuntimeError(
            "spec_decode CUDA extension not built. Run: pip install -e ."
        )

    k = draft_tokens.shape[0]
    V = target_logits.shape[1]

    _validate_inputs(target_logits, draft_logits, draft_tokens, k, V)

    # Apply temperature scaling.
    if temperature != 1.0:
        inv_temp = 1.0 / temperature
        target_logits = target_logits * inv_temp
        draft_logits  = draft_logits  * inv_temp

    # Generate random values for accept/reject decisions.
    random_values = torch.rand(k + 1, dtype=torch.float32, device=target_logits.device)

    batch_result = _ext.fused_verify_single(
        target_logits.contiguous(),
        draft_logits.contiguous(),
        draft_tokens.contiguous(),
        random_values,
    )

    # Unwrap batch dimension.
    return {
        "num_accepted": int(batch_result["num_accepted"][0].item()),
        "tokens":       batch_result["tokens"][0],
        "log_probs":    batch_result["log_probs"][0],
    }


def verify_speculative_tree(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    tree_topology: Dict[str, torch.Tensor],
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Run tree-structured speculative verification.

    Args:
        target_logits:  float16 or float32 tensor [node_count, vocab_size].
        draft_logits:   float16 or float32 tensor [node_count, vocab_size].
        tree_topology:  dict with:
            parent_ids:  int32 tensor [node_count] (root has parent -1)
            node_tokens: int32 tensor [node_count] draft token at each node
        temperature:    softmax temperature.

    Returns:
        dict with:
            path_length (int):                  length of accepted path
            path_tokens (Tensor [path_length+1]): accepted + correction token
            path_log_probs (Tensor [path_length+1])
    """
    raise NotImplementedError(
        "Tree verification Python binding not yet compiled. "
        "Build the extension and use _C directly for now."
    )


def _validate_inputs(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    k: int,
    V: int,
) -> None:
    if target_logits.shape != (k + 1, V):
        raise ValueError(
            f"target_logits must be [{k+1}, {V}], got {list(target_logits.shape)}"
        )
    if draft_logits.shape != (k, V):
        raise ValueError(
            f"draft_logits must be [{k}, {V}], got {list(draft_logits.shape)}"
        )
    if draft_tokens.shape != (k,):
        raise ValueError(
            f"draft_tokens must be [{k}], got {list(draft_tokens.shape)}"
        )
    if not target_logits.is_cuda:
        raise ValueError("target_logits must be a CUDA tensor")


__all__ = ["verify_speculative", "verify_speculative_tree"]
