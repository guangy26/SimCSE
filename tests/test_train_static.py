import ast
from pathlib import Path
import unittest


class TrainMainStaticTests(unittest.TestCase):
    def test_main_defines_results_before_returning_it(self):
        train_source = Path(__file__).resolve().parents[1] / "train.py"
        module = ast.parse(train_source.read_text())
        main_func = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        results_assignment_line = min(
            target.lineno
            for node in ast.walk(main_func)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "results"
        )
        results_return_line = min(
            node.lineno
            for node in ast.walk(main_func)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        )

        self.assertLess(results_assignment_line, results_return_line)


if __name__ == "__main__":
    unittest.main()
