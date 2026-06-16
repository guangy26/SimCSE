import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (REPO_ROOT / relative_path).read_text()


def parse_source(relative_path):
    return ast.parse(read_source(relative_path), filename=relative_path)


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_modules_do_not_import_sentence_transformers(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            tree = parse_source(relative_path)
            imported_modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.append(node.module)

            self.assertNotIn("sentence_transformers", imported_modules, relative_path)

    def test_train_main_initializes_results_before_returning_it(self):
        tree = parse_source("train.py")
        main_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        results_assign_index = None
        results_return_index = None
        for index, statement in enumerate(main_fn.body):
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name) and target.id == "results":
                        results_assign_index = index
            if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Name) and statement.value.id == "results":
                results_return_index = index

        self.assertIsNotNone(results_assign_index)
        self.assertIsNotNone(results_return_index)
        self.assertLess(results_assign_index, results_return_index)

    def test_train_helper_model_mask_is_frozen_raw_cosine_and_preserves_positives(self):
        source = read_source("train.py")

        self.assertIn("with torch.no_grad():", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad_(False)", source)
        self.assertIn("F.normalize(original_embeddings, p=2, dim=1)", source)
        self.assertIn("torch.matmul(original_embeddings, similar_embeddings.transpose(0, 1))", source)
        self.assertIn("similarity_mask[diag_idx, diag_idx] = 1.0", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_mlm_mask_tokens_is_implemented_without_mutating_contrastive_inputs(self):
        source = read_source("train.py")
        tree = parse_source("train.py")
        mask_tokens = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        )

        self.assertFalse(any(isinstance(statement, ast.Pass) for statement in mask_tokens.body))
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("masked_inputs = inputs.clone()", source)
        self.assertIn("return masked_inputs, labels", source)

    def test_trainer_imports_senteval_for_evaluation(self):
        tree = parse_source("simcse/trainers.py")
        imports_senteval = any(
            isinstance(node, ast.Import) and any(alias.name == "senteval" for alias in node.names)
            for node in ast.walk(tree)
        )

        self.assertTrue(imports_senteval)

    def test_model_expands_ddp_similarity_mask_and_keeps_diagonal_neutral(self):
        source = read_source("simcse/models.py")

        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = torch.ones_like(cos_sim)", source)
        self.assertIn("similarity_mask[diag_idx, diag_idx] = 1.0", source)


if __name__ == "__main__":
    unittest.main()
