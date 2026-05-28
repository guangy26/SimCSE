import importlib.util
from pathlib import Path

import torch


def load_masking_module():
    module_path = Path(__file__).resolve().parents[1] / "simcse" / "masking.py"
    spec = importlib.util.spec_from_file_location("masking", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_similarity_mask_uses_raw_cosine_and_keeps_diagonal_unmasked():
    masking = load_masking_module()
    original_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    similar_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    mask = masking.build_similarity_mask(
        original_embeddings,
        similar_embeddings,
        similarity_threshold_high=0.85,
        similarity_threshold_low=0.6,
    )

    assert mask[0, 0].item() == 1.0
    assert mask[1, 1].item() == 1.0
    assert mask[0, 1].item() == masking.MASKED_SIMILARITY_WEIGHT
    assert mask[1, 0].item() == 1.0


def test_assemble_global_similarity_mask_preserves_local_blocks_only():
    masking = load_masking_module()
    rank0_mask = torch.tensor([[1.0, 0.5], [0.25, 1.0]])
    rank1_mask = torch.tensor([[1.0, 0.75], [0.125, 1.0]])

    global_mask = masking.assemble_global_similarity_mask([rank0_mask, rank1_mask])

    assert torch.equal(global_mask[:2, :2], rank0_mask)
    assert torch.equal(global_mask[2:, 2:], rank1_mask)
    assert torch.equal(global_mask[:2, 2:], torch.ones(2, 2))
    assert torch.equal(global_mask[2:, :2], torch.ones(2, 2))
