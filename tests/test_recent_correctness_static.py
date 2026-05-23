import ast
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_repo_file(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_main_initializes_results_before_return(self):
        tree = ast.parse(read_repo_file("train.py"))
        main_fn = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        results_assignment = None
        results_return = None
        for node in ast.walk(main_fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "results":
                        results_assignment = node.lineno
            elif (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name)
                and node.value.id == "results"
            ):
                results_return = node.lineno

        self.assertIsNotNone(results_assignment)
        self.assertIsNotNone(results_return)
        self.assertLess(results_assignment, results_return)

    def test_no_unused_sentence_transformers_runtime_imports(self):
        train_source = read_repo_file("train.py")
        trainer_source = read_repo_file("simcse/trainers.py")

        self.assertNotIn("sentence_transformers", train_source)
        self.assertNotIn("sentence_transformers", trainer_source)

    def test_collator_mlm_masking_is_implemented_without_mutating_inputs(self):
        source = read_repo_file("train.py")

        self.assertIn("def mask_tokens(", source)
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return inputs, labels", source)
        self.assertNotRegex(source, r"def mask_tokens\([^)]*\):\s+pass")

    def test_senteval_is_imported_before_trainer_evaluate_uses_it(self):
        source = read_repo_file("simcse/trainers.py")

        import_line = source.index("import senteval")
        use_line = source.index("senteval.engine.SE")
        self.assertLess(import_line, use_line)

    def test_helper_model_mask_is_frozen_and_keeps_positive_pairs_neutral(self):
        source = read_repo_file("train.py")

        self.assertIn("help_model.eval()", source)
        self.assertIn("param.requires_grad = False", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("similarity_scores.fill_diagonal_(1.0)", source)

    def test_distributed_similarity_mask_expands_to_global_neutral_mask(self):
        source = read_repo_file("simcse/models.py")

        self.assertIn("dist.all_gather(tensor_list=gathered_masks", source)
        self.assertIn("global_similarity_mask = torch.ones_like(cos_sim)", source)
        self.assertIn("similarity_mask = global_similarity_mask", source)
        self.assertRegex(
            source,
            re.compile(
                r"global_similarity_mask\[row_start:row_start \+ rows, "
                r"col_start:col_start \+ cols\] = rank_mask"
            ),
        )


if __name__ == "__main__":
    unittest.main()
