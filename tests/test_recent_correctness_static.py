import ast
import pathlib
import re
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (REPO_ROOT / relative_path).read_text()


class RecentCorrectnessRegressionTests(unittest.TestCase):
    def test_train_main_always_returns_initialized_results(self):
        source = read_source("train.py")
        self.assertIn("results = {}", source)
        self.assertIn("if training_args.do_eval:", source)

        tree = ast.parse(source)
        main = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        result_assignments = [
            node for node in ast.walk(main)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        ]
        self.assertTrue(result_assignments, "main() must initialize results before returning it")

    def test_mlm_mask_tokens_is_implemented_and_non_mutating(self):
        source = read_source("train.py")
        mask_tokens_source = re.search(
            r"def mask_tokens\([\s\S]*?return inputs, labels",
            source,
        )
        self.assertIsNotNone(mask_tokens_source)
        mask_tokens_source = mask_tokens_source.group(0)

        self.assertNotIn("pass", mask_tokens_source)
        self.assertIn("inputs = inputs.clone()", mask_tokens_source)
        self.assertIn("labels = inputs.clone()", mask_tokens_source)
        self.assertIn("masked_indices", mask_tokens_source)
        self.assertIn("convert_tokens_to_ids(self.tokenizer.mask_token)", mask_tokens_source)
        self.assertIn("random_words", mask_tokens_source)

    def test_helper_similarity_mask_preserves_positive_pairs(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad = False", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("similarity_mask[positive_indices, positive_indices] = 1.0", source)
        self.assertNotIn("sim = Similarity(0.05)", source)

    def test_trainer_evaluation_flow_is_not_forcibly_disabled(self):
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
        self.assertIn("global_similarity_mask[start:end, start:end] = rank_similarity_mask", source)
        self.assertIn("similarity_mask = similarity_mask.to(device=cos_sim.device", source)

    def test_unused_sentence_transformers_dependency_is_not_imported(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            source = read_source(relative_path)
            self.assertNotIn("sentence_transformers", source)


if __name__ == "__main__":
    unittest.main()
