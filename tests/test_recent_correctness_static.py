import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (REPO_ROOT / relative_path).read_text()


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_main_initializes_results_before_return(self):
        tree = ast.parse(read_source("train.py"))
        main_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        results_assigned = False
        for node in ast.walk(main_func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "results":
                        results_assigned = True
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results":
                self.assertTrue(results_assigned)
                return

        self.fail("main() must return the initialized results dict")

    def test_mlm_masking_is_implemented_without_mutating_input_ids(self):
        tree = ast.parse(read_source("train.py"))
        mask_tokens = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens")

        self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens)))
        self.assertIn("inputs = inputs.clone()", ast.get_source_segment(read_source("train.py"), mask_tokens))
        self.assertTrue(any(isinstance(node, ast.Return) for node in ast.walk(mask_tokens)))

    def test_helper_model_mask_is_frozen_no_grad_and_preserves_positive_pairs(self):
        source = read_source("train.py")

        self.assertNotIn("from sentence_transformers import", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("diagonal = torch.eye", source)
        self.assertIn("mask_greater = mask_greater & ~diagonal", source)
        self.assertIn("mask_lower | diagonal", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("param.requires_grad = False", source)

    def test_senteval_is_imported_without_sentence_transformers_dependency(self):
        source = read_source("simcse/trainers.py")

        self.assertNotIn("from sentence_transformers import", source)
        self.assertIn("import senteval", source)

    def test_distributed_similarity_mask_expands_to_global_neutral_mask(self):
        source = read_source("simcse/models.py")

        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = similarity_mask.new_ones", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = local_similarity_mask", source)
        self.assertIn("similarity_mask = similarity_mask.to(cos_sim.device)", source)


if __name__ == "__main__":
    unittest.main()
