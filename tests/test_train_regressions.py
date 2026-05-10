import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class TrainMainRegressionTest(unittest.TestCase):
    def test_main_initializes_results_before_returning_it(self):
        tree = ast.parse((REPO_ROOT / "train.py").read_text())
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        result_return_lines = [
            node.lineno
            for node in ast.walk(main)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ]
        assignment_lines = [
            node.lineno
            for node in ast.walk(main)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "results"
        ]

        self.assertTrue(result_return_lines, "main() should return results")
        self.assertTrue(
            assignment_lines,
            "main() must initialize results even when evaluation is disabled",
        )
        self.assertLess(
            min(assignment_lines),
            min(result_return_lines),
            "results must be initialized before main() returns it",
        )


if __name__ == "__main__":
    unittest.main()
