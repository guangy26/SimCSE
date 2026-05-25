import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_entrypoint_returns_initialized_results(self):
        source = read_source("train.py")
        self.assertIn("results = {}\n    # TODO: Use our evaluation code", source)
        self.assertIn("return results", source)

    def test_no_undeclared_sentence_transformers_dependency(self):
        self.assertNotIn("sentence_transformers", read_source("train.py"))
        self.assertNotIn("sentence_transformers", read_source("simcse/trainers.py"))

    def test_mlm_masking_is_implemented_without_mutating_contrastive_inputs(self):
        tree = ast.parse(read_source("train.py"))
        mask_tokens = find_function(tree, "mask_tokens")
        self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens)))

        source = ast.get_source_segment(read_source("train.py"), mask_tokens)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return inputs, labels", source)

    def test_helper_mask_uses_raw_cosine_and_preserves_positive_diagonal(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertIn("parameter.requires_grad_(False)", source)

    def test_senteval_is_imported_before_trainer_evaluate_uses_it(self):
        source = read_source("simcse/trainers.py")
        self.assertIn("import senteval", source)
        self.assertIn("senteval.engine.SE", source)

    def test_distributed_similarity_mask_is_expanded_to_global_logits(self):
        source = read_source("simcse/models.py")
        self.assertIn("similarity_mask_list", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = torch.ones_like(cos_sim)", source)
        self.assertIn("similarity_mask.shape != cos_sim.shape", source)
        self.assertIn("similarity_mask.clamp_min(torch.finfo(cos_sim.dtype).tiny)", source)


if __name__ == "__main__":
    unittest.main()
