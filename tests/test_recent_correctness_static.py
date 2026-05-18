import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source_for(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_entrypoint_does_not_require_unused_sentence_transformers(self):
        self.assertNotIn("sentence_transformers", source_for("train.py"))
        self.assertNotIn("sentence_transformers", source_for("simcse/trainers.py"))

    def test_main_returns_initialized_results_when_eval_is_disabled(self):
        tree = ast.parse(source_for("train.py"))
        main_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        results_assignment = None
        return_results = None
        for node in ast.walk(main_fn):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
                and isinstance(node.value, ast.Dict)
            ):
                results_assignment = node
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name)
                and node.value.id == "results"
            ):
                return_results = node

        self.assertIsNotNone(results_assignment)
        self.assertIsNotNone(return_results)
        self.assertLess(results_assignment.lineno, return_results.lineno)

    def test_mlm_masking_is_implemented_without_mutating_contrastive_inputs(self):
        train_source = source_for("train.py")
        tree = ast.parse(train_source)
        mask_tokens = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        )
        body_source = ast.get_source_segment(train_source, mask_tokens)

        self.assertIn("inputs = inputs.clone()", body_source)
        self.assertIn("labels = inputs.clone()", body_source)
        self.assertIn("labels[~masked_indices] = -100", body_source)
        self.assertIn("return inputs, labels", body_source)
        self.assertNotIn("\n            pass", body_source)

    def test_helper_similarity_mask_uses_raw_cosine_and_preserves_positives(self):
        train_source = source_for("train.py")

        self.assertIn("with torch.no_grad():", train_source)
        self.assertIn("torch.nn.functional.normalize(original_embeddings", train_source)
        self.assertIn("torch.matmul(original_embeddings, similar_embeddings.transpose(0, 1))", train_source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", train_source)
        self.assertIn("parameter.requires_grad_(False)", train_source)
        self.assertNotIn("Similarity(0.05)", train_source)

    def test_distributed_similarity_mask_expands_to_global_logits(self):
        model_source = source_for("simcse/models.py")

        self.assertIn("gathered_masks = [torch.ones_like(local_mask)", model_source)
        self.assertIn("dist.all_gather(tensor_list=gathered_masks, tensor=local_mask)", model_source)
        self.assertIn("global_mask = torch.ones_like(cos_sim)", model_source)
        self.assertIn("if similarity_mask.shape != cos_sim.shape:", model_source)

    def test_senteval_is_imported_before_evaluate_uses_it(self):
        trainer_source = source_for("simcse/trainers.py")

        self.assertIn("import senteval", trainer_source)
        self.assertLess(trainer_source.index("import senteval"), trainer_source.index("senteval.engine.SE"))


if __name__ == "__main__":
    unittest.main()
