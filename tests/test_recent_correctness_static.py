import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_imports_do_not_require_sentence_transformers(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            with self.subTest(relative_path=relative_path):
                tree = ast.parse(read_repo_file(relative_path))
                for node in ast.walk(tree):
                    self.assertFalse(
                        isinstance(node, ast.ImportFrom) and node.module == "sentence_transformers",
                        f"{relative_path} must not import undeclared sentence_transformers at startup",
                    )
                    if isinstance(node, ast.Import):
                        imported_names = {alias.name for alias in node.names}
                        self.assertNotIn("sentence_transformers", imported_names)

    def test_train_main_returns_initialized_results(self):
        tree = ast.parse(read_repo_file("train.py"))
        main_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        result_assignments = [
            node.lineno
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "results"
        ]
        result_returns = [
            node.lineno
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results"
        ]
        self.assertTrue(result_assignments, "main() must initialize results before returning it")
        self.assertTrue(result_returns, "main() should continue returning the results mapping")
        self.assertLess(min(result_assignments), min(result_returns))

    def test_mlm_mask_tokens_is_implemented_without_mutating_inputs(self):
        source = read_repo_file("train.py")
        self.assertIn("def mask_tokens(", source)
        self.assertNotIn("def mask_tokens(\n            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None\n        ) -> Tuple[torch.Tensor, torch.Tensor]:\n            \"\"\"\n            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.\n            \"\"\"\n            pass", source)
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return inputs, labels", source)

    def test_helper_similarity_mask_uses_raw_cosine_positive_weights(self):
        source = read_repo_file("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("raw_similarity = F.cosine_similarity(", source)
        self.assertIn("similarity_mask = raw_similarity.new_ones", source)
        self.assertIn("similarity_mask[raw_similarity > self.similarity_threshold_high] = math.exp(-10)", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertNotIn("sim = Similarity(0.05)", source)

    def test_help_model_is_frozen_for_collator_masking(self):
        source = read_repo_file("train.py")
        self.assertIn("help_model.eval()", source)
        self.assertIn("for param in help_model.parameters():", source)
        self.assertIn("param.requires_grad_(False)", source)

    def test_distributed_similarity_mask_matches_global_logits(self):
        source = read_repo_file("simcse/models.py")
        self.assertIn("similarity_mask_list = [torch.ones_like(similarity_mask)", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list, tensor=similarity_mask.contiguous())", source)
        self.assertIn("global_similarity_mask = similarity_mask.new_ones", source)
        self.assertIn("if similarity_mask.shape != cos_sim.shape:", source)
        self.assertIn("similarity_mask = similarity_mask.clamp_min", source)

    def test_trainer_allows_callback_evaluation_and_imports_senteval(self):
        source = read_repo_file("simcse/trainers.py")
        self.assertIn("import senteval", source)
        self.assertNotIn("should_evaluate = False", source)
        self.assertNotIn("should_save = False", source)
        self.assertNotIn("should_save = True", source)


if __name__ == "__main__":
    unittest.main()
