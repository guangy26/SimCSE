import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


class RecentTrainingCorrectnessTests(unittest.TestCase):
    def test_train_returns_initialized_results(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        main_func = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        results_assignment_line = None
        return_results_line = None
        for node in ast.walk(main_func):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
                and isinstance(node.value, ast.Dict)
            ):
                results_assignment_line = node.lineno
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name)
                and node.value.id == "results"
            ):
                return_results_line = node.lineno

        self.assertIsNotNone(results_assignment_line)
        self.assertIsNotNone(return_results_line)
        self.assertLess(results_assignment_line, return_results_line)

    def test_mask_tokens_is_implemented_without_mutating_input_ids(self):
        source = read_source("train.py")
        mask_tokens_source = source[source.index("        def mask_tokens("):source.index("    if model_args.help_model_path")]

        self.assertNotIn("\n            pass\n", mask_tokens_source)
        self.assertIn("inputs = inputs.clone()", mask_tokens_source)
        self.assertIn("labels = inputs.clone()", mask_tokens_source)
        self.assertIn("labels[~masked_indices] = -100", mask_tokens_source)
        self.assertIn("return inputs, labels", mask_tokens_source)

    def test_helper_model_mask_uses_positive_weights_and_preserves_diagonal(self):
        source = read_source("train.py")
        collator_source = source[source.index("            # Compute similarity masks"):source.index("            if \"label\" in batch:")]

        self.assertIn("with torch.no_grad():", collator_source)
        self.assertIn("self.help_model.eval()", collator_source)
        self.assertIn("raw_similarity_scores = torch.cosine_similarity", collator_source)
        self.assertIn("similarity_mask = torch.ones_like(raw_similarity_scores)", collator_source)
        self.assertIn("high_similarity_negatives.fill_diagonal_(False)", collator_source)
        self.assertIn("masked_fill(high_similarity_negatives, math.exp(-10.0))", collator_source)
        self.assertNotIn("Similarity(0.05)", collator_source)

        load_source = source[source.index("    if model_args.help_model_path is not None:"):source.index("    data_collator =")]
        self.assertIn("help_model.eval()", load_source)
        self.assertIn("parameter.requires_grad = False", load_source)

    def test_similarity_mask_is_gathered_and_clamped_before_log(self):
        source = read_source("simcse/models.py")

        self.assertIn("mask_list = [torch.ones_like(similarity_mask) for _ in range(world_size)]", source)
        self.assertIn("dist.all_gather(tensor_list=mask_list, tensor=similarity_mask.contiguous())", source)
        self.assertIn("global_similarity_mask = torch.ones", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = process_mask", source)
        self.assertIn("similarity_mask = torch.clamp(similarity_mask, min=torch.finfo(cos_sim.dtype).tiny)", source)
        self.assertIn("cos_sim = cos_sim + torch.log(similarity_mask)", source)

    def test_trainer_imports_senteval_without_sentence_transformers(self):
        source = read_source("simcse/trainers.py")
        tree = ast.parse(source)
        imported_modules = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_from_modules = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }

        self.assertIn("senteval", imported_modules)
        self.assertNotIn("sentence_transformers", imported_modules)
        self.assertNotIn("sentence_transformers", imported_from_modules)


if __name__ == "__main__":
    unittest.main()
