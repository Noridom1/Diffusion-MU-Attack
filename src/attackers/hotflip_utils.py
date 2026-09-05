"""Small, dependency-light helpers for discrete HotFlip updates."""

from __future__ import annotations

import torch


def replacement_scores(embedding_grad, vocabulary_embeddings):
    """Return the first-order loss score for every position/token pair.

    Lower scores are preferred for a minimization objective.  The constant
    contribution of the current token is omitted because it does not affect
    ranking within a position.
    """

    if embedding_grad.ndim != 2 or vocabulary_embeddings.ndim != 2:
        raise ValueError("expected gradients [k, dim] and vocabulary [vocab, dim]")
    if embedding_grad.shape[1] != vocabulary_embeddings.shape[1]:
        raise ValueError("gradient and vocabulary embedding dimensions differ")
    return embedding_grad @ vocabulary_embeddings.transpose(0, 1)


def top_replacements(
    scores: torch.Tensor,
    current_ids: torch.Tensor,
    allowed_mask: torch.Tensor,
    topk: int,
):
    """Select the globally best ``topk`` position/token replacements."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [k, vocab]")
    if current_ids.numel() != scores.shape[0]:
        raise ValueError("current_ids length must equal the number of positions")
    if allowed_mask.shape != scores.shape:
        raise ValueError("allowed_mask must have the same shape as scores")
    if topk <= 0:
        raise ValueError("topk must be positive")

    masked_scores = scores.masked_fill(~allowed_mask, float("inf"))
    # Never propose keeping the current token at the same position.
    masked_scores = masked_scores.clone()
    masked_scores[torch.arange(scores.shape[0], device=scores.device), current_ids] = float("inf")
    flat_scores = masked_scores.flatten()
    valid = torch.isfinite(flat_scores)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return (
            torch.empty(0, dtype=torch.long, device=scores.device),
            torch.empty(0, dtype=torch.long, device=scores.device),
            torch.empty(0, dtype=scores.dtype, device=scores.device),
        )

    count = min(int(topk), valid_count)
    values, flat_indices = torch.topk(flat_scores, count, largest=False)
    vocab_size = scores.shape[1]
    positions = torch.div(flat_indices, vocab_size, rounding_mode='floor')
    token_ids = flat_indices.remainder(vocab_size)
    return positions, token_ids, values


def make_candidate_ids(current_ids: torch.Tensor, positions: torch.Tensor,
                       token_ids: torch.Tensor) -> torch.Tensor:
    """Create one discrete prefix per proposed replacement."""

    if current_ids.ndim != 1:
        raise ValueError("current_ids must have shape [k]")
    if positions.shape != token_ids.shape:
        raise ValueError("positions and token_ids must have the same shape")
    candidates = current_ids.unsqueeze(0).repeat(positions.numel(), 1)
    if positions.numel():
        candidates[torch.arange(positions.numel(), device=current_ids.device), positions] = token_ids
    return candidates
