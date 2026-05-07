import ast
from pathlib import Path


def test_main_initializes_results_before_returning_it():
    tree = ast.parse(Path("train.py").read_text())
    main_func = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    results_assignment_line = None
    return_line = None

    for node in ast.walk(main_func):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        ):
            results_assignment_line = node.lineno
        elif (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ):
            return_line = node.lineno

    assert results_assignment_line is not None
    assert return_line is not None
    assert results_assignment_line < return_line
