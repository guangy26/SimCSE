import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text()


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_main_initializes_results_before_returning(self):
        tree = ast.parse(read_source("train.py"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        saw_results_init = False
        for node in ast.walk(main):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
                and isinstance(node.value, ast.Dict)
            ):
                saw_results_init = True
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results":
                self.assertTrue(saw_results_init, "main() must initialize results before returning it")
                return
        self.fail("main() should return the evaluation results mapping")

    def test_train_eval_path_is_active_when_do_eval_is_set(self):
        source = read_source("train.py")
        self.assertIn("if training_args.do_eval:", source)
        self.assertIn("results = trainer.evaluate(eval_senteval_transfer=True)", source)

    def test_mlm_mask_tokens_is_implemented_and_non_mutating(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        mask_tokens = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens")
        self.assertFalse(any(isinstance(stmt, ast.Pass) for stmt in mask_tokens.body))
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("return inputs, labels", source)

    def test_helper_similarity_mask_uses_raw_cosine_and_preserves_positives(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity(", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_distributed_similarity_mask_is_expanded_to_global_batch(self):
        source = read_source("simcse/models.py")
        self.assertIn("similarity_mask_list", source)
        self.assertIn("global_similarity_mask", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)

    def test_senteval_is_imported_without_extra_sentence_transformers_dependency(self):
        trainer_source = read_source("simcse/trainers.py")
        train_source = read_source("train.py")
        self.assertIn("import senteval", trainer_source)
        self.assertNotIn("sentence_transformers", trainer_source)
        self.assertNotIn("sentence_transformers", train_source)


if __name__ == "__main__":
    unittest.main()
