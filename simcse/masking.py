import math

import torch
import torch.nn.functional as F


MASKED_SIMILARITY_WEIGHT = math.exp(-10)


def build_similarity_mask(
    original_embeddings: torch.Tensor,
    similar_embeddings: torch.Tensor,
    similarity_threshold_high: float,
    similarity_threshold_low: float,
) -> torch.Tensor:
    """Build loss weights from raw cosine similarities between batch pairs."""
    similarity_scores = F.cosine_similarity(
        original_embeddings.unsqueeze(1),
        similar_embeddings.unsqueeze(0),
        dim=-1,
    )
    similarity_mask = similarity_scores.clone()

    low_similarity = similarity_scores < similarity_threshold_low
    high_similarity = similarity_scores > similarity_threshold_high

    diagonal = torch.eye(
        similarity_scores.size(0),
        dtype=torch.bool,
        device=similarity_scores.device,
    )
    high_similarity = high_similarity & ~diagonal

    similarity_mask[low_similarity] = 1.0
    similarity_mask[high_similarity] = MASKED_SIMILARITY_WEIGHT
    similarity_mask[diagonal] = 1.0
    return similarity_mask


def assemble_global_similarity_mask(gathered_masks):
    """Expand per-rank square masks into the global batch mask used after all_gather."""
    if not gathered_masks:
        return None

    local_size = gathered_masks[0].size(0)
    world_size = len(gathered_masks)
    global_mask = gathered_masks[0].new_ones(
        (local_size * world_size, local_size * world_size)
    )
    for rank, local_mask in enumerate(gathered_masks):
        start = rank * local_size
        end = start + local_size
        global_mask[start:end, start:end] = local_mask
    return global_mask
