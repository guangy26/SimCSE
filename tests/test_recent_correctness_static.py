import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = (ROOT / "train.py").read_text()
MODELS_SOURCE = (ROOT / "simcse" / "models.py").read_text()
TRAINERS_SOURCE = (ROOT / "simcse" / "trainers.py").read_text()


def imported_modules(source):
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_entrypoint_only_imports_declared_dependencies(self):
        self.assertNotIn("sentence_transformers", imported_modules(TRAIN_SOURCE))
        self.assertNotIn("sentence_transformers", imported_modules(TRAINERS_SOURCE))

    def test_main_initializes_results_before_returning_it(self):
        tree = ast.parse(TRAIN_SOURCE)
        main_fn = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        assignments = [
            node.lineno
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        ]
        returns = [
            node.lineno
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ]

        self.assertTrue(assignments)
        self.assertTrue(returns)
        self.assertLess(min(assignments), min(returns))

    def test_senteval_is_imported_before_evaluate_uses_it(self):
        self.assertIn("senteval", imported_modules(TRAINERS_SOURCE))
        self.assertIn("senteval.engine.SE", TRAINERS_SOURCE)

    def test_mlm_masking_is_implemented_without_mutating_contrastive_inputs(self):
        self.assertIn("inputs = inputs.clone()", TRAIN_SOURCE)
        self.assertIn("labels = inputs.clone()", TRAIN_SOURCE)
        self.assertIn("labels[~masked_indices] = -100", TRAIN_SOURCE)
        self.assertIn("return inputs, labels", TRAIN_SOURCE)

    def test_helper_mask_is_inference_only_and_preserves_positive_pairs(self):
        self.assertIn("AutoModel.from_pretrained(model_args.help_model_path)", TRAIN_SOURCE)
        self.assertIn("help_model.eval()", TRAIN_SOURCE)
        self.assertIn("param.requires_grad_(False)", TRAIN_SOURCE)
        self.assertIn("with torch.no_grad():", TRAIN_SOURCE)
        self.assertIn("attention_mask=original_attention_mask", TRAIN_SOURCE)
        self.assertIn("torch.nn.functional.normalize", TRAIN_SOURCE)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", TRAIN_SOURCE)
        self.assertNotIn("sim = Similarity(0.05)", TRAIN_SOURCE)

    def test_distributed_mask_expands_to_global_logits_shape(self):
        self.assertIn("local_similarity_mask = similarity_mask.contiguous()", MODELS_SOURCE)
        self.assertIn(
            "dist.all_gather(tensor_list=similarity_mask_list, tensor=local_similarity_mask)",
            MODELS_SOURCE,
        )
        self.assertIn("global_similarity_mask = torch.ones_like(cos_sim)", MODELS_SOURCE)
        self.assertIn("similarity_mask = global_similarity_mask", MODELS_SOURCE)

    def test_trainer_does_not_override_callback_eval_and_save_decisions(self):
        self.assertNotIn("self.control.should_evaluate = False", TRAINERS_SOURCE)
        self.assertNotIn("self.control.should_save = False", TRAINERS_SOURCE)
        self.assertNotIn("self.control.should_save = True", TRAINERS_SOURCE)


if __name__ == "__main__":
    unittest.main()
