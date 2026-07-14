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
        elif isinstance(node, ast.ImportFrom) and node.module == module_name:
            return True
    return False


class RecentTrainingCorrectnessStaticTests(unittest.TestCase):
    def test_stale_sentence_transformers_imports_are_removed(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            with self.subTest(path=relative_path):
                self.assertFalse(
                    imports_module(parse_source(relative_path), "sentence_transformers")
                )

    def test_senteval_is_imported_where_evaluate_uses_it(self):
        self.assertTrue(imports_module(parse_source("simcse/trainers.py"), "senteval"))

    def test_train_main_defines_results_and_restores_evaluation(self):
        source = read_source("train.py")
        self.assertIn("results = {}", source)
        self.assertIn("if training_args.do_eval:", source)
        self.assertIn("results = trainer.evaluate(eval_senteval_transfer=True)", source)
        self.assertIn("return results", source)

    def test_mlm_masking_is_implemented_without_mutating_inputs(self):
        source = read_source("train.py")
        functions = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        ]
        self.assertEqual(len(functions), 1)
        function_source = ast.get_source_segment(source, functions[0])
        self.assertNotIn("pass", function_source)
        self.assertIn("inputs = inputs.clone()", function_source)
        self.assertIn("labels = inputs.clone()", function_source)
        self.assertIn("labels[~masked_indices] = -100", function_source)
        self.assertIn("return inputs, labels", function_source)

    def test_helper_mask_uses_raw_cosine_and_preserves_positive_pairs(self):
        source = read_source("train.py")
        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad = False", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("attention_mask=attention_mask", source)
        self.assertIn("Similarity(1.0)", source)
        self.assertIn("similarity_mask[diagonal, diagonal] = 1.0", source)

    def test_distributed_mask_expands_to_global_neutral_blocks(self):
        source = read_source("simcse/models.py")
        self.assertIn("dist.all_gather(", source)
        self.assertIn("tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = similarity_mask.new_ones(", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = rank_mask", source)
        self.assertIn("similarity_mask[diagonal, diagonal] = 1.0", source)
        self.assertIn("similarity_mask.clamp_min(1e-6)", source)

    def test_senteval_paths_do_not_depend_on_current_working_directory(self):
        source = read_source("simcse/trainers.py")
        self.assertIn("os.path.dirname(os.path.dirname(os.path.abspath(__file__)))", source)
        self.assertIn("PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL, 'data')", source)

    def test_trainer_does_not_override_callback_eval_and_save_decisions(self):
        source = read_source("simcse/trainers.py")
        self.assertNotIn("self.control.should_evaluate =", source)
        self.assertNotIn("self.control.should_save =", source)


if __name__ == "__main__":
    unittest.main()
