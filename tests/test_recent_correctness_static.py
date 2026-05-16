import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text()


def parse_source(relative_path):
    return ast.parse(read_source(relative_path))


def find_function(module, name):
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"could not find function {name}")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_main_returns_initialized_results(self):
        module = parse_source("train.py")
        main = find_function(module, "main")

        result_assignments = [
            node.lineno
            for node in ast.walk(main)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "results"
        ]
        result_returns = [
            node.lineno
            for node in ast.walk(main)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "results"
        ]

        self.assertTrue(result_assignments, "main() must initialize results before returning it")
        self.assertTrue(result_returns, "main() should return the results dictionary")
        self.assertLess(min(result_assignments), max(result_returns))

    def test_mlm_mask_tokens_is_implemented_without_mutating_input_ids(self):
        module = parse_source("train.py")
        mask_tokens = find_function(module, "mask_tokens")
        source_segment = ast.get_source_segment(read_source("train.py"), mask_tokens)

        self.assertFalse(
            any(isinstance(node, ast.Pass) for node in ast.walk(mask_tokens)),
            "mask_tokens() must not be left as a pass stub",
        )
        self.assertIn("inputs = inputs.clone()", source_segment)
        self.assertIn("labels = inputs.clone()", source_segment)
        self.assertIn("get_special_tokens_mask", source_segment)
        self.assertIn("torch.bernoulli", source_segment)
        self.assertIn("return inputs, labels", source_segment)

    def test_helper_similarity_mask_does_not_penalize_positive_pairs(self):
        source = read_source("train.py")

        self.assertIn("F.cosine_similarity", source)
        self.assertIn("positive_pairs = torch.eye", source)
        self.assertIn("mask_greater = (similarity_scores > self.similarity_threshold_high) & ~positive_pairs", source)
        self.assertIn("similarity_mask[mask_lower | positive_pairs] = 1.0", source)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("parameter.requires_grad_(False)", source)

    def test_senteval_is_imported_before_trainer_evaluation_uses_it(self):
        module = parse_source("simcse/trainers.py")
        imported_names = {
            alias.asname or alias.name
            for node in module.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        self.assertIn("senteval", imported_names)

    def test_distributed_similarity_mask_is_expanded_to_global_logits(self):
        source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list", source)
        self.assertIn("global_similarity_mask", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn(
            "global_similarity_mask[block_start:block_end, block_start:block_end] = rank_similarity_mask",
            source,
        )
        self.assertIn("similarity_mask = similarity_mask.to(device=cos_sim.device", source)


if __name__ == "__main__":
    unittest.main()
