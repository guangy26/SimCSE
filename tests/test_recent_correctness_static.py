import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"
TRAINERS = ROOT / "simcse" / "trainers.py"
MODELS = ROOT / "simcse" / "models.py"


def read(path):
    return path.read_text(encoding="utf-8")


def function_source(path, name):
    source = read(path)
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if not matches:
        raise AssertionError(f"{name} not found in {path}")
    return ast.get_source_segment(source, matches[0])


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_entrypoints_do_not_require_sentence_transformers(self):
        combined_source = read(TRAIN) + "\n" + read(TRAINERS)
        self.assertNotIn("sentence_transformers", combined_source)

    def test_train_main_returns_initialized_results_for_train_only_runs(self):
        source = read(TRAIN)
        self.assertIn("results = {}", source)
        self.assertLess(source.index("results = {}"), source.index("return results"))

    def test_mlm_mask_tokens_is_implemented_without_mutating_inputs(self):
        mask_tokens = function_source(TRAIN, "mask_tokens")
        self.assertNotIn("pass", mask_tokens)
        self.assertIn("labels = inputs.clone()", mask_tokens)
        self.assertIn("inputs = inputs.clone()", mask_tokens)
        self.assertIn("labels[~masked_indices] = -100", mask_tokens)
        self.assertIn("convert_tokens_to_ids(self.tokenizer.mask_token)", mask_tokens)
        self.assertIn("return inputs, labels", mask_tokens)

    def test_helper_similarity_mask_uses_frozen_raw_cosine_and_preserves_positives(self):
        source = read(TRAIN)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad = False", source)
        self.assertIn("torch.nn.CosineSimilarity(dim=-1)", source)
        self.assertIn("attention_mask=original_attention_mask", source)
        self.assertIn("attention_mask=similar_attention_mask", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_senteval_is_imported_before_trainer_evaluation_uses_it(self):
        source = read(TRAINERS)
        tree = ast.parse(source)
        imported_modules = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        self.assertIn("senteval", imported_modules)
        self.assertLess(source.index("import senteval"), source.index("senteval.engine.SE"))

    def test_distributed_similarity_mask_expands_to_global_batch(self):
        source = read(MODELS)
        self.assertIn("global_similarity_mask", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = rank_similarity_mask", source)
        self.assertIn("similarity_mask = similarity_mask.clamp_min", source)


if __name__ == "__main__":
    unittest.main()
