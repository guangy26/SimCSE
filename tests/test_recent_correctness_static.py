import ast
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text()


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_main_always_returns_defined_results(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        main_fn = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        assigned_results = any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
            for node in ast.walk(main_fn)
        )
        returns_results = any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
            for node in ast.walk(main_fn)
        )

        self.assertTrue(assigned_results)
        self.assertTrue(returns_results)

    def test_no_undeclared_sentence_transformers_startup_dependency(self):
        self.assertNotIn("sentence_transformers", read_source("train.py"))
        self.assertNotIn("sentence_transformers", read_source("simcse/trainers.py"))

    def test_senteval_is_imported_for_evaluation(self):
        source = read_source("simcse/trainers.py")
        self.assertIn("import senteval", source)
        self.assertIn("senteval.engine.SE", source)

    def test_trainer_does_not_force_disable_configured_evaluation(self):
        source = read_source("simcse/trainers.py")
        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)

    def test_mlm_masking_is_implemented_without_mutating_input_ids(self):
        source = read_source("train.py")
        mask_tokens = re.search(
            r"def mask_tokens\([\s\S]+?^\s{4}if model_args\.help_model_path",
            source,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(mask_tokens)
        body = mask_tokens.group(0)
        self.assertNotIn("pass", body)
        self.assertIn("inputs = inputs.clone()", body)
        self.assertIn("labels[~masked_indices] = -100", body)
        self.assertIn("return inputs, labels", body)

    def test_helper_similarity_mask_preserves_positives_and_avoids_grad_graphs(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("help_model.requires_grad_(False)", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("similarity_mask[diagonal, diagonal] = 1.0", source)

    def test_ddp_similarity_mask_is_expanded_to_global_logits(self):
        source = read_source("simcse/models.py")
        self.assertIn("similarity_mask_list", source)
        self.assertIn("dist.all_gather", source)
        self.assertIn("global_similarity_mask = torch.ones_like(cos_sim)", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = rank_similarity_mask", source)
        self.assertIn("cos_sim = cos_sim + torch.log(similarity_mask)", source)


if __name__ == "__main__":
    unittest.main()
