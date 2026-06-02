import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text()


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_main_returns_initialized_results(self):
        tree = ast.parse(read_source("train.py"))
        main_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        results_assignment_line = None
        return_results_line = None
        for node in ast.walk(main_fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "results":
                        results_assignment_line = node.lineno
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name)
                and node.value.id == "results"
            ):
                return_results_line = node.lineno

        self.assertIsNotNone(results_assignment_line)
        self.assertIsNotNone(return_results_line)
        self.assertLess(results_assignment_line, return_results_line)

    def test_senteval_is_imported_where_evaluate_uses_it(self):
        source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", source)
        self.assertIn("senteval.engine.SE", source)
        self.assertNotIn("sentence_transformers", source)

    def test_mlm_masking_is_implemented_without_mutating_contrastive_inputs(self):
        source = read_source("train.py")

        self.assertIn("def mask_tokens", source)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return inputs, labels", source)
        self.assertNotIn("sentence_transformers", source)

    def test_helper_similarity_mask_preserves_positives_and_freezes_helper_model(self):
        source = read_source("train.py")

        self.assertIn("AutoModel.from_pretrained(model_args.help_model_path)", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("param.requires_grad = False", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("similarity_mask = torch.ones_like(similarity_scores)", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)

    def test_distributed_similarity_mask_expands_to_global_neutral_mask(self):
        source = read_source("simcse/models.py")

        self.assertIn("local_similarity_mask = similarity_mask.contiguous()", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list, tensor=local_similarity_mask)", source)
        self.assertIn("global_similarity_mask = torch.ones_like(cos_sim)", source)
        self.assertIn("similarity_mask = global_similarity_mask", source)


if __name__ == "__main__":
    unittest.main()
