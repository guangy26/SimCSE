import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path):
    return (REPO_ROOT / relative_path).read_text()


def _function_source(relative_path, function_name):
    source = _source(relative_path)
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{function_name} not found in {relative_path}")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_main_initializes_results_when_evaluation_is_disabled(self):
        main_source = _function_source("train.py", "main")
        main_ast = ast.parse(main_source)

        assignments = [
            node
            for node in ast.walk(main_ast)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        ]

        self.assertTrue(assignments, "main() must define results before returning it")
        self.assertLess(main_source.rfind("results = {}"), main_source.rfind("return results"))

    def test_distributed_similarity_mask_is_expanded_with_gathered_logits(self):
        helper_source = _function_source("simcse/models.py", "_gather_distributed_similarity_mask")
        cl_forward_source = _function_source("simcse/models.py", "cl_forward")

        self.assertIn("dist.all_gather", helper_source)
        self.assertIn("torch.block_diag", helper_source)
        self.assertLess(
            cl_forward_source.find("_gather_distributed_similarity_mask(similarity_mask)"),
            cl_forward_source.find("torch.log(similarity_mask)"),
        )


if __name__ == "__main__":
    unittest.main()
