import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Could not find function {name}")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_script_has_no_stale_sentence_transformers_import(self):
        source = read_source("train.py")
        self.assertNotIn("sentence_transformers", source)

    def test_train_main_initializes_results_before_return(self):
        tree = ast.parse(read_source("train.py"))
        main_node = find_function(tree, "main")
        result_assign_lines = [
            node.lineno
            for node in ast.walk(main_node)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "results"
        ]
        return_lines = [
            node.lineno
            for node in ast.walk(main_node)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ]
        self.assertTrue(result_assign_lines)
        self.assertTrue(return_lines)
        self.assertLess(min(result_assign_lines), max(return_lines))

    def test_mlm_masking_is_implemented_without_mutating_input_ids(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        mask_tokens = find_function(tree, "mask_tokens")
        self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens)))
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return inputs, labels", source)

    def test_helper_mask_uses_raw_cosine_and_preserves_positive_pairs(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad_(False)", source)
        self.assertIn("torch.nn.functional.cosine_similarity", source)
        self.assertIn("similarity_mask = torch.ones_like(similarity_scores)", source)
        self.assertIn("similarity_mask[diagonal] = 1.0", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_distributed_similarity_mask_is_expanded_to_global_logits(self):
        source = read_source("simcse/models.py")
        self.assertIn("similarity_mask_list = [torch.ones_like(similarity_mask)", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn(
            "global_similarity_mask = similarity_mask.new_ones((z1.size(0), z2.size(0)))",
            source,
        )
        self.assertIn("similarity_mask = global_similarity_mask", source)

    def test_senteval_is_imported_without_stale_optional_dependency(self):
        source = read_source("simcse/trainers.py")
        self.assertIn("import senteval", source)
        self.assertIn("senteval.engine.SE", source)
        self.assertNotIn("sentence_transformers", source)

    def test_trainer_honors_callback_evaluation_and_save_decisions(self):
        source = read_source("simcse/trainers.py")
        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)
        self.assertNotIn("self.control.should_save = True", source)


if __name__ == "__main__":
    unittest.main()
