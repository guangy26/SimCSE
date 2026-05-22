import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text()


def parse_source(relative_path):
    return ast.parse(read_source(relative_path))


def imported_modules(tree):
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_entrypoint_returns_initialized_results(self):
        tree = parse_source("train.py")
        main = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        assigned_results = any(
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

        self.assertTrue(assigned_results)
        self.assertTrue(returns_results)

    def test_mlm_mask_tokens_is_implemented_without_mutating_inputs(self):
        tree = parse_source("train.py")
        mask_tokens = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        )
        statements = [
            node for node in ast.walk(mask_tokens)
            if isinstance(node, (ast.Pass, ast.Return, ast.Call, ast.Assign))
        ]

        self.assertFalse(any(isinstance(node, ast.Pass) for node in statements))
        self.assertTrue(any(isinstance(node, ast.Return) for node in statements))
        self.assertIn("inputs.clone", ast.unparse(mask_tokens))

    def test_helper_mask_uses_raw_cosine_no_grad_and_preserves_positive_diagonal(self):
        source = read_source("train.py")

        self.assertIn("with torch.no_grad()", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("similarity_scores[diagonal_indices, diagonal_indices] = 1.0", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_no_stale_sentence_transformers_import_on_training_path(self):
        train_imports = imported_modules(parse_source("train.py"))
        trainer_imports = imported_modules(parse_source("simcse/trainers.py"))

        self.assertNotIn("sentence_transformers", train_imports)
        self.assertNotIn("sentence_transformers", trainer_imports)
        self.assertIn("senteval", trainer_imports)

    def test_distributed_similarity_mask_matches_gathered_logits(self):
        source = read_source("simcse/models.py")

        self.assertIn("dist.all_gather(tensor_list=mask_list", source)
        self.assertIn("expanded_mask = torch.ones_like(cos_sim)", source)
        self.assertIn("similarity_mask.clamp_min(torch.finfo(cos_sim.dtype).tiny)", source)


if __name__ == "__main__":
    unittest.main()
