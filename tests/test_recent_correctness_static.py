import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_main_always_returns_initialized_results(self):
        source = read_source("train.py")
        tree = ast.parse(source)

        main_func = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        assignments = [
            node for node in ast.walk(main_func)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        ]
        returns = [
            node for node in ast.walk(main_func)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ]

        self.assertTrue(assignments, "main() must initialize results before returning it")
        self.assertTrue(returns, "main() should continue returning the results dictionary")

    def test_no_import_of_undeclared_sentence_transformers_dependency(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            with self.subTest(path=relative_path):
                self.assertNotIn("sentence_transformers", read_source(relative_path))

    def test_mlm_masking_is_implemented_and_non_mutating(self):
        source = read_source("train.py")

        self.assertNotIn("def mask_tokens(\n", source.split("pass", 1)[0])
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return inputs, labels", source)

    def test_helper_similarity_mask_preserves_positive_pairs_without_gradients(self):
        source = read_source("train.py")

        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("help_model.requires_grad_(False)", source)

    def test_trainer_uses_senteval_and_honors_callback_control(self):
        source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", source)
        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)
        self.assertNotIn("self.control.should_save = True", source)

    def test_distributed_similarity_mask_expands_to_global_batch(self):
        source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask", source)
        self.assertIn("similarity_mask = similarity_mask.to(device=cos_sim.device, dtype=cos_sim.dtype)", source)


if __name__ == "__main__":
    unittest.main()
