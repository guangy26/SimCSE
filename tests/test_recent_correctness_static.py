import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def parse_source(relative_path):
    return ast.parse(read_source(relative_path), filename=relative_path)


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Could not find function {name}")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_modules_do_not_import_missing_sentence_transformers(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            with self.subTest(relative_path=relative_path):
                self.assertNotIn("sentence_transformers", read_source(relative_path))

    def test_train_main_initializes_results_before_return(self):
        main_fn = find_function(parse_source("train.py"), "main")
        return_lines = [
            node.lineno
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ]
        result_assign_lines = [
            node.lineno
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "results"
        ]

        self.assertTrue(return_lines, "main() should keep returning results")
        self.assertTrue(result_assign_lines, "main() must initialize results on train-only runs")
        self.assertLess(min(result_assign_lines), min(return_lines))

    def test_mlm_mask_tokens_is_implemented_without_mutating_input_ids(self):
        train_tree = parse_source("train.py")
        mask_tokens = find_function(train_tree, "mask_tokens")
        self.assertFalse(
            any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens)),
            "mask_tokens must not be a pass stub",
        )

        source = ast.get_source_segment(read_source("train.py"), mask_tokens)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("convert_tokens_to_ids", source)

    def test_helper_similarity_mask_uses_raw_cosine_and_preserves_positives(self):
        source = read_source("train.py")

        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad = False", source)
        self.assertNotIn("Similarity(", source)

    def test_trainer_keeps_senteval_import_and_callback_control(self):
        source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", source)
        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)

    def test_distributed_similarity_mask_expands_to_global_shape(self):
        source = read_source("simcse/models.py")

        self.assertIn("dist.all_gather(tensor_list=mask_list", source)
        self.assertIn("global_mask = similarity_mask.new_ones", source)
        self.assertIn("similarity_mask = global_mask", source)
        self.assertIn("cos_sim = cos_sim + torch.log(similarity_mask)", source)


if __name__ == "__main__":
    unittest.main()
