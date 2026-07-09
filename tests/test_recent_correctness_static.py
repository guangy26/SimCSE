import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def parse_source(relative_path):
    return ast.parse(read_source(relative_path))


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_main_initializes_results_before_return(self):
        tree = parse_source("train.py")
        main_func = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        results_assign_lines = [
            node.lineno
            for node in ast.walk(main_func)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        ]
        return_results_lines = [
            node.lineno
            for node in ast.walk(main_func)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ]

        self.assertTrue(results_assign_lines)
        self.assertTrue(return_results_lines)
        self.assertLess(min(results_assign_lines), min(return_results_lines))

    def test_similarity_mask_uses_raw_cosine_and_preserves_diagonal(self):
        source = read_source("train.py")

        self.assertIn("torch.nn.functional.cosine_similarity", source)
        self.assertNotIn("sim = Similarity(0.05)", source)
        self.assertNotIn("similarity_scores = sim(", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("diagonal_mask = torch.eye", source)
        self.assertIn("& ~diagonal_mask", source)
        self.assertIn("similarity_mask[diagonal_mask] = 1.0", source)
        self.assertIn("help_model.requires_grad_(False)", source)

    def test_mlm_masking_is_implemented(self):
        source = read_source("train.py")

        self.assertNotIn("def mask_tokens(\n            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None\n        ) -> Tuple[torch.Tensor, torch.Tensor]:\n            \"\"\"\n            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.\n            \"\"\"\n            pass", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids", source)
        self.assertIn("random_words = torch.randint", source)
        self.assertIn("return inputs, labels", source)

    def test_distributed_similarity_mask_expands_to_global_logits_shape(self):
        source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list = [torch.ones_like(local_similarity_mask)", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("similarity_mask = torch.ones(", source)
        self.assertIn("similarity_mask[row_start:row_end, col_start:col_end] = rank_similarity_mask", source)
        self.assertIn("similarity_mask.shape != cos_sim.shape", source)
        self.assertIn("similarity_mask must contain only positive values", source)

    def test_no_undeclared_sentence_transformers_imports_remain(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            tree = parse_source(relative_path)
            imported_modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported_modules.append(node.module)

            self.assertNotIn("sentence_transformers", imported_modules, relative_path)

    def test_senteval_is_imported_for_trainer_evaluate(self):
        tree = parse_source("simcse/trainers.py")

        imported_modules = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        imported_modules.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )

        self.assertIn("senteval", imported_modules)


if __name__ == "__main__":
    unittest.main()
