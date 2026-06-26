import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_and_trainer_do_not_require_sentence_transformers(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            tree = ast.parse(read_source(relative_path))
            imported_modules = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            self.assertNotIn("sentence_transformers", imported_modules)

    def test_train_main_returns_initialized_results(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        main_func = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        results_assigned = any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
            for node in ast.walk(main_func)
        )
        results_returned = any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
            for node in ast.walk(main_func)
        )
        self.assertTrue(results_assigned)
        self.assertTrue(results_returned)

    def test_senteval_is_imported_before_trainer_evaluate_uses_it(self):
        source = read_source("simcse/trainers.py")
        self.assertIn("import senteval", source)
        self.assertIn("senteval.engine.SE", source)

    def test_mlm_masking_is_implemented_without_mutating_input_ids(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        mask_tokens = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        )
        self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens)))
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("return inputs, labels", source)

    def test_helper_similarity_mask_preserves_positive_pairs_and_freezes_model(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad_(False)", source)
        self.assertIn("diagonal = torch.eye", source)
        self.assertIn("similarity_mask[diagonal] = 1.0", source)
        self.assertIn("torch.nn.functional.cosine_similarity", source)

    def test_distributed_similarity_mask_expands_to_global_neutral_mask(self):
        source = read_source("simcse/models.py")
        self.assertIn("dist.all_gather(tensor_list=mask_list", source)
        self.assertIn("global_similarity_mask = torch.ones", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = process_mask", source)
        self.assertIn("similarity_mask = global_similarity_mask", source)

    def test_trainer_does_not_force_disable_callback_eval_or_save(self):
        source = read_source("simcse/trainers.py")
        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)
        self.assertNotIn("self.control.should_save = True", source)


if __name__ == "__main__":
    unittest.main()
