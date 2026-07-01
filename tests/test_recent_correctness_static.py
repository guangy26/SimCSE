import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def parse_source(relative_path):
    return ast.parse(read_source(relative_path))


def imports_module(tree, module_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == module_name for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == module_name:
                return True
    return False


class RecentTrainingCorrectnessStaticTests(unittest.TestCase):
    def test_stale_sentence_transformers_imports_are_removed(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            with self.subTest(path=relative_path):
                self.assertFalse(imports_module(parse_source(relative_path), "sentence_transformers"))

    def test_senteval_is_imported_where_evaluate_uses_it(self):
        self.assertTrue(imports_module(parse_source("simcse/trainers.py"), "senteval"))

    def test_train_main_initializes_and_populates_results(self):
        train_source = read_source("train.py")
        self.assertIn("results = {}", train_source)
        self.assertIn("if training_args.do_eval:", train_source)
        self.assertIn("results = trainer.evaluate(eval_senteval_transfer=True)", train_source)
        self.assertIn("return results", train_source)

    def test_mlm_masking_is_implemented_without_mutating_inputs(self):
        tree = parse_source("train.py")
        mask_tokens = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        ]
        self.assertEqual(len(mask_tokens), 1)
        self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens[0])))

        mask_source = ast.get_source_segment(read_source("train.py"), mask_tokens[0])
        self.assertIn("labels = inputs.clone()", mask_source)
        self.assertIn("inputs = inputs.clone()", mask_source)
        self.assertIn("labels[~masked_indices] = -100", mask_source)
        self.assertIn("torch.randint", mask_source)

    def test_help_model_mask_preserves_positives_and_avoids_gradients(self):
        train_source = read_source("train.py")
        self.assertIn("help_model.eval()", train_source)
        self.assertIn("parameter.requires_grad_(False)", train_source)
        self.assertIn("with torch.no_grad():", train_source)
        self.assertIn("torch.nn.functional.cosine_similarity", train_source)
        self.assertIn("similarity_mask[diagonal, diagonal] = 1.0", train_source)

    def test_distributed_similarity_mask_expands_to_global_neutral_mask(self):
        models_source = read_source("simcse/models.py")
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", models_source)
        self.assertIn("new_ones((global_batch_size, global_batch_size))", models_source)
        self.assertIn("global_similarity_mask[start:end, start:end] = rank_similarity_mask", models_source)
        self.assertIn("similarity_mask.shape != cos_sim.shape", models_source)
        self.assertIn("clamp_min(torch.finfo(cos_sim.dtype).tiny)", models_source)

    def test_trainer_respects_callback_evaluation_and_save_decisions(self):
        trainers_source = read_source("simcse/trainers.py")
        self.assertNotIn("self.control.should_evaluate = False", trainers_source)
        self.assertNotIn("self.control.should_save = False", trainers_source)
        self.assertNotIn("self.control.should_save = True", trainers_source)


if __name__ == "__main__":
    unittest.main()
