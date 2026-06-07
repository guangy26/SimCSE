import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_imports_only_declared_runtime_dependencies(self):
        train_source = read_source("train.py")
        trainers_source = read_source("simcse/trainers.py")

        self.assertNotIn("sentence_transformers", train_source)
        self.assertNotIn("sentence_transformers", trainers_source)

    def test_train_returns_initialized_results(self):
        train_source = read_source("train.py")
        tree = ast.parse(train_source)
        main_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        result_assignments = [
            node for node in ast.walk(main_fn)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        ]
        result_returns = [
            node for node in ast.walk(main_fn)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ]

        self.assertTrue(result_assignments, "main() must initialize results before returning it")
        self.assertTrue(result_returns, "main() should continue returning the results mapping")
        self.assertLess(result_assignments[0].lineno, result_returns[-1].lineno)

    def test_helper_similarity_mask_preserves_positive_pairs(self):
        train_source = read_source("train.py")

        self.assertIn("with torch.no_grad():", train_source)
        self.assertIn("help_model.eval()", train_source)
        self.assertIn("param.requires_grad_(False)", train_source)
        self.assertIn("attention_mask=original_attention_mask", train_source)
        self.assertIn("attention_mask=similar_attention_mask", train_source)
        self.assertIn("torch.cosine_similarity", train_source)
        self.assertIn("similarity_scores.fill_diagonal_(1.0)", train_source)

    def test_mlm_masking_is_implemented_without_mutating_inputs(self):
        train_source = read_source("train.py")
        tree = ast.parse(train_source)
        mask_tokens = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        ][0]

        self.assertFalse(
            any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens)),
            "mask_tokens must not be a stub",
        )
        mask_tokens_source = ast.get_source_segment(train_source, mask_tokens)
        self.assertIn("inputs = inputs.clone()", mask_tokens_source)
        self.assertIn("return inputs, labels", mask_tokens_source)

    def test_senteval_and_trainer_evaluation_flow_are_enabled(self):
        trainers_source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", trainers_source)
        self.assertNotIn("should_evaluate = False", trainers_source)
        self.assertNotIn("should_save = False", trainers_source)
        self.assertNotIn("should_save = True", trainers_source)

    def test_distributed_similarity_mask_expands_to_global_logits(self):
        models_source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list", models_source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", models_source)
        self.assertIn("global_similarity_mask = torch.ones", models_source)
        self.assertIn("row_start:row_start + rows", models_source)
        self.assertIn("col_start:col_start + cols", models_source)
        self.assertIn("similarity_mask.clamp_min(torch.finfo(cos_sim.dtype).tiny)", models_source)


if __name__ == "__main__":
    unittest.main()
