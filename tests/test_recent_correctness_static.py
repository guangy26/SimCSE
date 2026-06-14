import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_main_initializes_results_before_return(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        main_fn = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        results_assign_line = None
        return_results_line = None
        for node in ast.walk(main_fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "results":
                        results_assign_line = node.lineno
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name)
                and node.value.id == "results"
            ):
                return_results_line = node.lineno

        self.assertIsNotNone(results_assign_line)
        self.assertIsNotNone(return_results_line)
        self.assertLess(results_assign_line, return_results_line)

    def test_mlm_mask_tokens_is_implemented_without_mutating_inputs(self):
        source = read_source("train.py")
        self.assertIn("def mask_tokens(", source)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("convert_tokens_to_ids(self.tokenizer.mask_token)", source)

    def test_helper_similarity_mask_uses_raw_no_grad_scores_and_keeps_positives(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity(", source)
        self.assertIn("similarity_scores.fill_diagonal_(1.0)", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad = False", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_distributed_similarity_mask_expands_to_global_block_mask(self):
        source = read_source("simcse/models.py")
        self.assertIn("similarity_mask_list", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = similarity_mask.new_ones", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = rank_similarity_mask", source)
        self.assertIn("cos_sim = cos_sim + torch.log(similarity_mask)", source)

    def test_senteval_import_restored_without_sentence_transformers_dependency(self):
        trainer_source = read_source("simcse/trainers.py")
        train_source = read_source("train.py")
        self.assertIn("import senteval", trainer_source)
        self.assertNotIn("sentence_transformers", trainer_source)
        self.assertNotIn("sentence_transformers", train_source)


if __name__ == "__main__":
    unittest.main()
