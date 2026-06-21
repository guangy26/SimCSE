import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def imported_modules(source):
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_modules_do_not_require_sentence_transformers_at_startup(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            modules = imported_modules(read_source(relative_path))
            self.assertFalse(
                any(module == "sentence_transformers" or module.startswith("sentence_transformers.") for module in modules),
                f"{relative_path} imports optional sentence_transformers at startup",
            )

    def test_senteval_is_imported_before_trainer_evaluate_uses_it(self):
        source = read_source("simcse/trainers.py")
        self.assertIn("import senteval", source)
        self.assertLess(source.index("import senteval"), source.index("senteval.engine.SE"))

    def test_main_returns_initialized_results(self):
        tree = ast.parse(read_source("train.py"))
        main_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        assigns_results = [
            node
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        ]
        returns_results = [
            node
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results"
        ]
        self.assertTrue(assigns_results)
        self.assertTrue(returns_results)
        self.assertLess(assigns_results[0].lineno, returns_results[-1].lineno)

    def test_mlm_mask_tokens_is_implemented_without_mutating_inputs(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        mask_tokens = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        ][0]
        self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens)))
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertTrue(any(isinstance(node, ast.Return) for node in ast.walk(mask_tokens)))

    def test_helper_similarity_mask_preserves_positive_pairs(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("similarity_mask[similarity_scores > self.similarity_threshold_high] = math.exp(-10)", source)
        self.assertIn("diag = torch.arange(similarity_mask.size(0)", source)
        self.assertIn("similarity_mask[diag, diag] = 1.0", source)

    def test_distributed_similarity_mask_is_expanded_to_global_logits(self):
        source = read_source("simcse/models.py")
        self.assertIn("similarity_mask_list", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("similarity_mask_global = torch.ones", source)
        self.assertIn("similarity_mask_global[start:end, start:end] = rank_similarity_mask", source)

    def test_trainer_does_not_override_callback_eval_save_decisions(self):
        source = read_source("simcse/trainers.py")
        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)
        self.assertNotIn("self.control.should_save = True", source)


if __name__ == "__main__":
    unittest.main()
