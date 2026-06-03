import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_SOURCE = (REPO_ROOT / "train.py").read_text()
TRAINERS_SOURCE = (REPO_ROOT / "simcse" / "trainers.py").read_text()
MODELS_SOURCE = (REPO_ROOT / "simcse" / "models.py").read_text()


def imported_modules(source):
    tree = ast.parse(source)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_entrypoint_has_no_undeclared_sentence_transformers_import(self):
        self.assertNotIn("sentence_transformers", imported_modules(TRAIN_SOURCE))
        self.assertNotIn("sentence_transformers", imported_modules(TRAINERS_SOURCE))

    def test_train_main_initializes_results_before_return(self):
        tree = ast.parse(TRAIN_SOURCE)
        main_func = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        result_assign_line = None
        return_line = None
        for node in ast.walk(main_func):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
                and isinstance(node.value, ast.Dict)
            ):
                result_assign_line = node.lineno
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name)
                and node.value.id == "results"
            ):
                return_line = node.lineno

        self.assertIsNotNone(result_assign_line)
        self.assertIsNotNone(return_line)
        self.assertLess(result_assign_line, return_line)

    def test_senteval_is_imported_before_evaluate_uses_it(self):
        self.assertIn("senteval", imported_modules(TRAINERS_SOURCE))
        self.assertIn("senteval.engine.SE", TRAINERS_SOURCE)

    def test_mlm_mask_tokens_is_implemented_and_non_mutating(self):
        self.assertIn("inputs = inputs.clone()", TRAIN_SOURCE)
        self.assertIn("labels = inputs.clone()", TRAIN_SOURCE)
        self.assertIn("masked_indices = torch.bernoulli(probability_matrix).bool()", TRAIN_SOURCE)
        self.assertIn("return inputs, labels", TRAIN_SOURCE)
        self.assertNotIn('def mask_tokens(\n            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None\n        ) -> Tuple[torch.Tensor, torch.Tensor]:\n            """\n            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.\n            """\n            pass', TRAIN_SOURCE)

    def test_helper_similarity_mask_preserves_positives_and_does_not_track_gradients(self):
        self.assertIn("with torch.no_grad():", TRAIN_SOURCE)
        self.assertIn("help_model.eval()", TRAIN_SOURCE)
        self.assertIn("param.requires_grad_(False)", TRAIN_SOURCE)
        self.assertIn("AutoModel.from_pretrained(model_args.help_model_path)", TRAIN_SOURCE)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", TRAIN_SOURCE)
        self.assertIn("torch.nn.functional.normalize", TRAIN_SOURCE)
        self.assertNotIn("sim = Similarity(0.05)", TRAIN_SOURCE)

    def test_distributed_similarity_mask_matches_gathered_logits_shape(self):
        self.assertIn("similarity_mask_list = [torch.ones_like(similarity_mask) for _ in range(world_size)]", MODELS_SOURCE)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", MODELS_SOURCE)
        self.assertIn("global_similarity_mask = similarity_mask.new_ones", MODELS_SOURCE)
        self.assertIn("global_similarity_mask[start:end, start:end] = rank_similarity_mask", MODELS_SOURCE)
        self.assertIn("similarity_mask = global_similarity_mask", MODELS_SOURCE)


if __name__ == "__main__":
    unittest.main()
