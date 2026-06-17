import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_modules_do_not_import_missing_sentence_transformers(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            with self.subTest(path=relative_path):
                tree = ast.parse(read_source(relative_path))
                imported_modules = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_modules.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module is not None:
                        imported_modules.append(node.module)

                self.assertNotIn("sentence_transformers", imported_modules)

    def test_train_main_initializes_results_before_returning(self):
        tree = ast.parse(read_source("train.py"))
        main_func = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        results_assignment = None
        results_return = None
        for node in ast.walk(main_func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "results":
                        results_assignment = node.lineno
            elif (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name)
                and node.value.id == "results"
            ):
                results_return = node.lineno

        self.assertIsNotNone(results_assignment)
        self.assertIsNotNone(results_return)
        self.assertLess(results_assignment, results_return)

    def test_mlm_mask_tokens_is_implemented_without_mutating_inputs(self):
        source = read_source("train.py")

        self.assertIn("def mask_tokens(", source)
        self.assertNotIn("def mask_tokens(\n            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None\n        ) -> Tuple[torch.Tensor, torch.Tensor]:\n            \"\"\"\n            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.\n            \"\"\"\n            pass", source)
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("masked_inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return masked_inputs, labels", source)

    def test_helper_similarity_mask_preserves_positives_and_uses_raw_cosine(self):
        source = read_source("train.py")

        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity(", source)
        self.assertIn("similarity_mask = similarity_scores.clone()", source)
        self.assertIn("similarity_mask[positive_indices, positive_indices] = 1.0", source)
        self.assertIn("parameter.requires_grad = False", source)
        self.assertNotIn("sim = Similarity(0.05)", source)

    def test_distributed_similarity_masks_expand_to_global_logits_shape(self):
        source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list = [torch.ones_like(similarity_mask)", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = torch.ones(", source)
        self.assertIn("global_similarity_mask[process_slice, process_slice] = process_similarity_mask", source)
        self.assertIn("similarity_mask = global_similarity_mask", source)
        self.assertIn("cos_sim = cos_sim + torch.log(similarity_mask)", source)

    def test_senteval_is_imported_before_trainer_evaluate_uses_it(self):
        source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", source)
        self.assertLess(source.index("import senteval"), source.index("senteval.engine.SE"))


if __name__ == "__main__":
    unittest.main()
