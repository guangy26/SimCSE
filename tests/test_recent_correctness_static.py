import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text()


def parse_source(relative_path):
    return ast.parse(read_source(relative_path))


def imported_modules(tree):
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Could not find function {name}")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_modules_do_not_require_unused_sentence_transformers_dependency(self):
        self.assertNotIn("sentence_transformers", imported_modules(parse_source("train.py")))
        self.assertNotIn("sentence_transformers", imported_modules(parse_source("simcse/trainers.py")))

    def test_train_main_initializes_results_before_returning_it(self):
        main_fn = find_function(parse_source("train.py"), "main")
        result_assignments = [
            node for node in ast.walk(main_fn)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        ]
        returns_results = [
            node for node in ast.walk(main_fn)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results"
        ]

        self.assertTrue(result_assignments, "main() must initialize results before returning it")
        self.assertTrue(returns_results, "main() should return the results dictionary")

    def test_mlm_mask_tokens_is_implemented_without_mutating_contrastive_inputs(self):
        source = read_source("train.py")
        mask_tokens = find_function(parse_source("train.py"), "mask_tokens")

        self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens)))
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("return inputs, labels", source)

    def test_helper_similarity_mask_uses_frozen_raw_cosine_and_preserves_positives(self):
        source = read_source("train.py")

        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad = False", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("F.cosine_similarity", source)
        self.assertNotIn("Similarity(0.05)", source)
        self.assertIn("diagonal_mask = torch.eye", source)
        self.assertIn("similarity_scores[diagonal_mask] = 1.0", source)

    def test_trainer_can_import_and_run_senteval_evaluation_callbacks(self):
        source = read_source("simcse/trainers.py")
        tree = parse_source("simcse/trainers.py")

        self.assertIn("senteval", imported_modules(tree))
        self.assertNotIn("self.control.should_evaluate = False", source)

    def test_distributed_similarity_mask_is_gathered_to_match_global_logits(self):
        source = read_source("simcse/models.py")

        self.assertIn("dist.all_gather(tensor_list=mask_list", source)
        self.assertIn("global_similarity_mask = torch.ones_like(cos_sim)", source)
        self.assertIn("similarity_mask.shape != cos_sim.shape", source)
        self.assertIn("torch.log(similarity_mask)", source)


if __name__ == "__main__":
    unittest.main()
