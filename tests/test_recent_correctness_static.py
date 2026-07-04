import ast
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN = (ROOT / "train.py").read_text()
MODELS = (ROOT / "simcse" / "models.py").read_text()
TRAINERS = (ROOT / "simcse" / "trainers.py").read_text()


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_entrypoint_does_not_return_undefined_results(self):
        tree = ast.parse(TRAIN)
        main = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        has_results_init = any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
            for node in ast.walk(main)
        )
        returns_results = any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
            for node in ast.walk(main)
        )

        self.assertTrue(has_results_init)
        self.assertTrue(returns_results)

    def test_imports_match_declared_dependencies_and_senteval_usage(self):
        for source, filename in [(TRAIN, "train.py"), (TRAINERS, "simcse/trainers.py")]:
            tree = ast.parse(source, filename=filename)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertNotIn("sentence_transformers", imports)

        self.assertIn("import senteval", TRAINERS)

    def test_helper_similarity_mask_uses_raw_padded_cosine_and_keeps_positives(self):
        self.assertIn("torch.no_grad()", TRAIN)
        self.assertIn("help_model.eval()", TRAIN)
        self.assertIn("param.requires_grad_(False)", TRAIN)
        self.assertIn("F.cosine_similarity", TRAIN)
        self.assertIn("_masked_mean_pool", TRAIN)
        self.assertRegex(TRAIN, r"similarity_mask\[diag_idx,\s*diag_idx\]\s*=\s*1\.0")
        self.assertNotRegex(TRAIN, r"sim\s*=\s*Similarity\(0\.05\)")

    def test_distributed_similarity_mask_expands_to_global_shape(self):
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", MODELS)
        self.assertIn("global_similarity_mask", MODELS)
        self.assertRegex(
            MODELS,
            re.compile(
                r"global_similarity_mask\[start:end,\s*start:end\]\s*=\s*rank_similarity_mask"
            ),
        )

    def test_trainer_does_not_force_disable_evaluation_or_step_saves(self):
        self.assertNotIn("self.control.should_evaluate = False", TRAINERS)
        self.assertNotIn("self.control.should_save = False", TRAINERS)

    def test_mlm_masking_returns_inputs_and_labels_without_mutating_source(self):
        self.assertIn("inputs = inputs.clone()", TRAIN)
        self.assertIn("labels = inputs.clone()", TRAIN)
        self.assertIn("return inputs, labels", TRAIN)
        self.assertNotIn("\n            pass\n", TRAIN)


if __name__ == "__main__":
    unittest.main()
