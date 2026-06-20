import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_does_not_import_undeclared_sentence_transformers(self):
        self.assertNotIn("sentence_transformers", read_source("train.py"))
        self.assertNotIn("sentence_transformers", read_source("simcse/trainers.py"))

    def test_main_initializes_results_before_returning_it(self):
        tree = ast.parse(read_source("train.py"))
        main_function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        assigned_before_return = False
        for node in ast.walk(main_function):
            if isinstance(node, ast.Assign):
                assigned_before_return |= any(
                    isinstance(target, ast.Name) and target.id == "results"
                    for target in node.targets
                )
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results":
                self.assertTrue(assigned_before_return)
                return
        self.fail("main() must return the initialized results variable")

    def test_mask_tokens_is_implemented_without_mutating_contrastive_inputs(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        mask_tokens = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        ][0]
        self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens)))
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return inputs, labels", source)

    def test_helper_similarity_mask_uses_raw_cosine_and_preserves_positives(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("F.cosine_similarity", source)
        self.assertNotIn("Similarity(0.05)", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertIn("parameter.requires_grad = False", source)

    def test_distributed_similarity_mask_is_expanded_before_logit_addition(self):
        source = read_source("simcse/models.py")
        self.assertIn("similarity_mask_list", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = torch.ones_like(cos_sim)", source)
        self.assertIn("similarity_mask.shape != cos_sim.shape", source)

    def test_senteval_is_imported_where_trainer_evaluate_uses_it(self):
        source = read_source("simcse/trainers.py")
        self.assertIn("import senteval", source)
        self.assertIn("senteval.engine.SE", source)


if __name__ == "__main__":
    unittest.main()
