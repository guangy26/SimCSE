import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text()


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_entrypoint_has_no_undeclared_optional_imports(self):
        train_source = read_source("train.py")
        trainers_source = read_source("simcse/trainers.py")

        self.assertNotIn("sentence_transformers", train_source)
        self.assertNotIn("sentence_transformers", trainers_source)

    def test_train_main_initializes_and_returns_results(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        main_node = find_function(tree, "main")
        main_source = ast.get_source_segment(source, main_node)

        self.assertIn("results = {}", main_source)
        self.assertIn("if training_args.do_eval:", main_source)
        self.assertIn("results = trainer.evaluate(eval_senteval_transfer=True)", main_source)
        self.assertIn("return results", main_source)

    def test_mlm_mask_tokens_is_implemented_without_mutating_inputs(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        mask_tokens_node = find_function(tree, "mask_tokens")
        mask_tokens_source = ast.get_source_segment(source, mask_tokens_node)

        self.assertNotIn("pass", mask_tokens_source)
        self.assertIn("labels = inputs.clone()", mask_tokens_source)
        self.assertIn("inputs = inputs.clone()", mask_tokens_source)
        self.assertIn("labels[~masked_indices] = -100", mask_tokens_source)
        self.assertIn("return inputs, labels", mask_tokens_source)

    def test_helper_similarity_mask_preserves_positive_pairs(self):
        source = read_source("train.py")

        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity(", source)
        self.assertIn("positive_pairs = torch.eye", source)
        self.assertIn("similarity_mask = similarity_mask.masked_fill(positive_pairs, 1.0)", source)
        self.assertIn("parameter.requires_grad_(False)", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_trainer_evaluation_is_imported_and_not_forcibly_disabled(self):
        source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", source)
        self.assertIn("se = senteval.engine.SE", source)
        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)

    def test_distributed_similarity_mask_matches_gathered_logits(self):
        source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list", source)
        self.assertIn("dist.all_gather(\n                tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = local_similarity_mask.new_ones", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = process_similarity_mask", source)
        self.assertIn("similarity_mask = similarity_mask.to(device=cos_sim.device", source)


if __name__ == "__main__":
    unittest.main()
