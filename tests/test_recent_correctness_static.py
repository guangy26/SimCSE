import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class CriticalTrainingRegressionTests(unittest.TestCase):
    def test_train_main_initializes_results_before_return(self):
        tree = ast.parse(read_source("train.py"))
        main_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        result_assign_lines = [
            node.lineno
            for node in ast.walk(main_func)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "results"
        ]
        result_return_lines = [
            node.lineno
            for node in ast.walk(main_func)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ]

        self.assertTrue(result_return_lines, "main() should return the training/eval results object")
        self.assertTrue(result_assign_lines, "results must be initialized even when evaluation is disabled")
        self.assertLess(min(result_assign_lines), min(result_return_lines))

    def test_no_startup_dependency_on_sentence_transformers(self):
        self.assertNotIn("sentence_transformers", read_source("train.py"))
        self.assertNotIn("sentence_transformers", read_source("simcse/trainers.py"))

    def test_senteval_is_imported_for_evaluate(self):
        trainers_source = read_source("simcse/trainers.py")
        self.assertIn("import senteval", trainers_source)
        self.assertIn("senteval.engine.SE", trainers_source)

    def test_mlm_mask_tokens_is_implemented_without_mutating_inputs(self):
        source = read_source("train.py")
        tree = ast.parse(source)
        mask_tokens_func = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "mask_tokens"
        )

        self.assertIn("def mask_tokens(", source)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("get_special_tokens_mask", source)
        self.assertIn("return inputs, labels", source)
        self.assertFalse(any(isinstance(node, ast.Pass) for node in mask_tokens_func.body))

    def test_helper_similarity_mask_preserves_positive_diagonal(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("torch.eye(mask_greater.size(0)", source)
        self.assertIn("mask_greater = mask_greater.masked_fill(diagonal, False)", source)
        self.assertIn("similarity_mask = torch.ones_like(similarity_scores)", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("param.requires_grad = False", source)

    def test_distributed_similarity_mask_expands_to_global_shape(self):
        source = read_source("simcse/models.py")
        self.assertIn("similarity_mask_list", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = similarity_mask.new_ones", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = local_similarity_mask", source)
        self.assertIn("torch.log(similarity_mask.to(cos_sim.device))", source)


if __name__ == "__main__":
    unittest.main()
