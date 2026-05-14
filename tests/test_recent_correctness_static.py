import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_module(relative_path):
    return ast.parse((ROOT / relative_path).read_text(), filename=relative_path)


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found")


def test_train_main_initializes_returned_results():
    tree = parse_module("train.py")
    main = find_function(tree, "main")

    assigns_results = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        and isinstance(node.value, ast.Dict)
        for node in ast.walk(main)
    )
    returns_results = any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "results"
        for node in ast.walk(main)
    )

    assert assigns_results
    assert returns_results


def test_mask_tokens_is_implemented_without_mutating_contrastive_inputs():
    tree = parse_module("train.py")
    mask_tokens = find_function(tree, "mask_tokens")

    assert not any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens))
    clones_input = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "inputs" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "clone"
        for node in ast.walk(mask_tokens)
    )
    returns_inputs_and_labels = any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and [elt.id for elt in node.value.elts if isinstance(elt, ast.Name)] == ["inputs", "labels"]
        for node in ast.walk(mask_tokens)
    )

    assert clones_input
    assert returns_inputs_and_labels


def test_trainer_imports_senteval_before_evaluate_uses_it():
    tree = parse_module("simcse/trainers.py")
    imports_senteval = any(
        isinstance(node, ast.Import)
        and any(alias.name == "senteval" for alias in node.names)
        for node in tree.body
    )
    evaluate = find_function(tree, "evaluate")
    uses_senteval = any(isinstance(node, ast.Name) and node.id == "senteval" for node in ast.walk(evaluate))

    assert imports_senteval
    assert uses_senteval


def test_distributed_similarity_mask_is_gathered_and_shape_checked():
    tree = parse_module("simcse/models.py")
    helper = find_function(tree, "gather_distributed_similarity_mask")
    cl_forward = find_function(tree, "cl_forward")

    helper_uses_all_gather = any(
        isinstance(node, ast.Attribute) and node.attr == "all_gather" for node in ast.walk(helper)
    )
    helper_starts_from_neutral_mask = any(
        isinstance(node, ast.Attribute) and node.attr == "new_ones" for node in ast.walk(helper)
    )
    cl_forward_calls_helper = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "gather_distributed_similarity_mask"
        for node in ast.walk(cl_forward)
    )
    cl_forward_checks_shape = any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "ValueError"
        for node in ast.walk(cl_forward)
    )

    assert helper_uses_all_gather
    assert helper_starts_from_neutral_mask
    assert cl_forward_calls_helper
    assert cl_forward_checks_shape
