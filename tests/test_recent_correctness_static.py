import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_main_returns_initialized_results_and_runs_guarded_eval(self):
        source = read_source("train.py")

        self.assertIn("results = {}", source)
        self.assertIn("if training_args.do_eval:", source)
        self.assertIn("results = trainer.evaluate(eval_senteval_transfer=True)", source)
        self.assertIn("return results", source)

    def test_mlm_masking_is_implemented_without_mutating_contrastive_inputs(self):
        source = read_source("train.py")
        mask_tokens_body = re.search(
            r"def mask_tokens\([^)]*\).*?(?=\n    if model_args\.help_model_path)",
            source,
            flags=re.S,
        )
        self.assertIsNotNone(mask_tokens_body)
        body = mask_tokens_body.group(0)

        self.assertNotIn("pass", body)
        self.assertIn("inputs = inputs.clone()", body)
        self.assertIn("labels = inputs.clone()", body)
        self.assertIn("labels[~masked_indices] = -100", body)
        self.assertIn("return inputs, labels", body)

    def test_helper_similarity_mask_preserves_positive_pairs_and_uses_raw_cosine(self):
        source = read_source("train.py")

        self.assertIn("parameter.requires_grad_(False)", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("attention_mask=original_attention_mask", source)
        self.assertIn("torch.nn.functional.normalize", source)
        self.assertIn("torch.matmul(original_embeddings, similar_embeddings.transpose(0, 1))", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_distributed_similarity_mask_expands_to_global_logits(self):
        source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list = [torch.ones_like(similarity_mask)", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = torch.ones", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = local_similarity_mask", source)
        self.assertIn("torch.log(similarity_mask.to(cos_sim.device))", source)

    def test_senteval_is_imported_and_trainer_control_is_not_forced_off(self):
        source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", source)
        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)

    def test_training_imports_do_not_require_unused_sentence_transformers(self):
        self.assertNotIn("sentence_transformers", read_source("train.py"))
        self.assertNotIn("sentence_transformers", read_source("simcse/trainers.py"))


if __name__ == "__main__":
    unittest.main()
