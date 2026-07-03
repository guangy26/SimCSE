import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def parse_source(relative_path):
    return ast.parse(read_source(relative_path), filename=relative_path)


class RecentTrainingCorrectnessTests(unittest.TestCase):
    def test_training_modules_do_not_import_unconfigured_sentence_transformers(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            with self.subTest(relative_path=relative_path):
                tree = parse_source(relative_path)
                imported_modules = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_modules.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported_modules.append(node.module)

                self.assertNotIn("sentence_transformers", imported_modules)

    def test_train_main_returns_initialized_results(self):
        tree = parse_source("train.py")
        main_func = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        saw_results_assignment = False
        saw_results_return = False
        for node in ast.walk(main_func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "results":
                        saw_results_assignment = True
            elif isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results":
                saw_results_return = True

        self.assertTrue(saw_results_assignment)
        self.assertTrue(saw_results_return)

    def test_mlm_mask_tokens_is_implemented_without_mutating_inputs(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        mask_tokens_methods = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        ]
        self.assertEqual(len(mask_tokens_methods), 1)
        self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens_methods[0])))
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return inputs, labels", source)

    def test_helper_similarity_mask_does_not_train_helper_or_mask_positives(self):
        source = read_source("train.py")

        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad_(False)", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("raw_similarity = F.cosine_similarity", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_trainer_evaluation_callbacks_and_senteval_import_are_active(self):
        source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", source)
        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)
        self.assertNotIn("self.control.should_save = True", source)

    def test_distributed_similarity_mask_expands_to_global_logits(self):
        source = read_source("simcse/models.py")

        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = torch.ones", source)
        self.assertIn("similarity_mask = similarity_mask.to(device=cos_sim.device", source)
        self.assertIn("similarity_mask.clamp_min(torch.finfo(cos_sim.dtype).tiny)", source)
        self.assertIn("cos_sim = cos_sim + torch.log(similarity_mask)", source)


if __name__ == "__main__":
    unittest.main()
