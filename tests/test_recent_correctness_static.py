import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text()


def parse_source(relative_path):
    return ast.parse(read_source(relative_path))


def imported_modules(tree):
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Could not find function {name}")


def test_training_modules_do_not_require_unused_sentence_transformers_dependency():
    assert "sentence_transformers" not in imported_modules(parse_source("train.py"))
    assert "sentence_transformers" not in imported_modules(parse_source("simcse/trainers.py"))


def test_train_main_initializes_results_before_returning_it():
    main_fn = find_function(parse_source("train.py"), "main")
    result_assignments = [
        node for node in ast.walk(main_fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
    ]
    returns_results = [
        node for node in ast.walk(main_fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results"
    ]

    assert result_assignments, "main() must initialize results before returning it"
    assert returns_results, "main() should return the results dictionary"


def test_mlm_mask_tokens_is_implemented_without_mutating_contrastive_inputs():
    source = read_source("train.py")
    mask_tokens = find_function(parse_source("train.py"), "mask_tokens")

    assert not any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens))
    assert "inputs = inputs.clone()" in source
    assert "return inputs, labels" in source


def test_helper_similarity_mask_uses_frozen_raw_cosine_and_preserves_positives():
    source = read_source("train.py")

    assert "help_model.eval()" in source
    assert "parameter.requires_grad = False" in source
    assert "with torch.no_grad():" in source
    assert "F.cosine_similarity" in source
    assert "Similarity(0.05)" not in source
    assert "diagonal_mask = torch.eye" in source
    assert "similarity_scores[diagonal_mask] = 1.0" in source


def test_trainer_can_import_and_run_senteval_evaluation_callbacks():
    source = read_source("simcse/trainers.py")
    tree = parse_source("simcse/trainers.py")

    assert "senteval" in imported_modules(tree)
    assert "self.control.should_evaluate = False" not in source


def test_distributed_similarity_mask_is_gathered_to_match_global_logits():
    source = read_source("simcse/models.py")

    assert "dist.all_gather(tensor_list=mask_list" in source
    assert "global_similarity_mask = torch.ones_like(cos_sim)" in source
    assert "similarity_mask.shape != cos_sim.shape" in source
    assert "torch.log(similarity_mask)" in source
