import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_runtime_files_do_not_import_undeclared_sentence_transformers(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            with self.subTest(relative_path=relative_path):
                tree = ast.parse(read_source(relative_path))
                imports = [
                    node
                    for node in ast.walk(tree)
                    if (
                        isinstance(node, ast.ImportFrom)
                        and node.module == "sentence_transformers"
                    )
                    or (
                        isinstance(node, ast.Import)
                        and any(alias.name == "sentence_transformers" for alias in node.names)
                    )
                ]
                self.assertEqual(imports, [])

    def test_train_main_initializes_results_before_returning_it(self):
        tree = ast.parse(read_source("train.py"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        result_assignments = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
            and isinstance(node.value, ast.Dict)
        ]
        returns_results = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ]

        self.assertTrue(result_assignments)
        self.assertTrue(returns_results)
        self.assertLess(result_assignments[0].lineno, returns_results[-1].lineno)

    def test_mlm_mask_tokens_is_implemented_without_mutating_inputs(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        mask_tokens = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        )
        segment = ast.get_source_segment(source, mask_tokens)

        self.assertFalse(len(mask_tokens.body) == 1 and isinstance(mask_tokens.body[0], ast.Pass))
        self.assertIn("inputs = inputs.clone()", segment)
        self.assertIn("labels = inputs.clone()", segment)
        self.assertIn("labels[~masked_indices] = -100", segment)
        self.assertIn("return inputs, labels", segment)

    def test_helper_similarity_mask_does_not_penalize_positive_pairs(self):
        source = read_source("train.py")

        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad_(False)", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("raw_similarity = F.cosine_similarity", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_distributed_loss_expands_local_similarity_masks_to_global_shape(self):
        source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = rank_similarity_mask", source)
        self.assertIn("similarity_mask.clamp_min(torch.finfo(cos_sim.dtype).tiny)", source)

    def test_senteval_is_imported_before_trainer_evaluation_uses_it(self):
        source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", source)
        self.assertIn("senteval.engine.SE", source)


if __name__ == "__main__":
    unittest.main()
