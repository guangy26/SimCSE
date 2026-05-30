import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def parse_source(relative_path):
    return ast.parse(read_source(relative_path), filename=relative_path)


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_entrypoint_does_not_require_sentence_transformers(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            with self.subTest(relative_path=relative_path):
                tree = parse_source(relative_path)
                imports = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module == "sentence_transformers"
                ]
                self.assertEqual(imports, [], f"{relative_path} imports undeclared sentence_transformers")

    def test_main_initializes_results_before_returning(self):
        tree = parse_source("train.py")
        main = find_function(tree, "main")

        result_assignments = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
        ]
        result_returns = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results"
        ]

        self.assertTrue(result_assignments, "main() must initialize results before returning it")
        self.assertTrue(result_returns, "main() should still return results")
        self.assertLess(
            min(node.lineno for node in result_assignments),
            min(node.lineno for node in result_returns),
        )

    def test_mask_tokens_restores_mlm_behavior_without_mutating_inputs(self):
        train_source = read_source("train.py")
        tree = ast.parse(train_source)
        mask_tokens = find_function(tree, "mask_tokens")
        mask_tokens_source = ast.get_source_segment(train_source, mask_tokens)

        self.assertNotIsInstance(mask_tokens.body[0], ast.Pass)
        self.assertIn("inputs = inputs.clone()", mask_tokens_source)
        self.assertIn("labels[~masked_indices] = -100", mask_tokens_source)
        self.assertIn("convert_tokens_to_ids(self.tokenizer.mask_token)", mask_tokens_source)
        self.assertIn("return inputs, labels", mask_tokens_source)

    def test_helper_similarity_mask_is_neutral_for_positives_and_frozen(self):
        train_source = read_source("train.py")

        self.assertIn("AutoModel.from_pretrained(model_args.help_model_path)", train_source)
        self.assertIn("help_model.eval()", train_source)
        self.assertIn("parameter.requires_grad_(False)", train_source)
        self.assertIn("with torch.no_grad():", train_source)
        self.assertIn("F.cosine_similarity", train_source)
        self.assertIn("similarity_mask = torch.ones_like(similarity_scores)", train_source)
        self.assertIn("non_diagonal = ~torch.eye", train_source)
        self.assertIn("(similarity_scores > self.similarity_threshold_high) & non_diagonal", train_source)
        self.assertNotIn("Similarity(0.05)", train_source)

    def test_trainer_evaluation_callbacks_and_senteval_are_enabled(self):
        trainer_source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", trainer_source)
        self.assertNotIn("should_evaluate = False", trainer_source)
        self.assertNotIn("should_save = False", trainer_source)

    def test_distributed_similarity_mask_is_expanded_to_global_logits(self):
        model_source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list", model_source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", model_source)
        self.assertIn("global_similarity_mask = torch.ones", model_source)
        self.assertIn("global_similarity_mask[start:end, start:end] = rank_similarity_mask", model_source)
        self.assertIn("clamp_min(torch.finfo(cos_sim.dtype).tiny)", model_source)


if __name__ == "__main__":
    unittest.main()
