import ast
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_imports_only_declared_dependencies(self):
        source = read_source("train.py")
        tree = ast.parse(source)

        forbidden_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                forbidden_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                forbidden_imports.append(node.module)

        self.assertNotIn("sentence_transformers", forbidden_imports)

    def test_train_main_returns_initialized_results(self):
        source = read_source("train.py")

        self.assertRegex(source, r"\n\s*results = \{\}\n")
        self.assertRegex(source, r"\n\s*return results\n")

    def test_mask_tokens_is_implemented_without_mutating_inputs(self):
        source = read_source("train.py")

        self.assertNotRegex(source, r"def mask_tokens\([^)]*\).*?\n\s*pass\n", re.DOTALL)
        self.assertIn("labels = inputs.clone()", source)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return inputs, labels", source)

    def test_helper_similarity_mask_is_neutral_for_positive_pairs(self):
        source = read_source("train.py")

        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.cosine_similarity", source)
        self.assertIn("similarity_mask = torch.ones_like(similarity_scores)", source)
        self.assertIn("diagonal = torch.eye(bs", source)
        self.assertIn("similarity_mask[mask_greater & ~diagonal] = math.exp(-10)", source)
        self.assertIn("parameter.requires_grad_(False)", source)

    def test_trainer_imports_senteval_and_respects_callback_control(self):
        source = read_source("simcse/trainers.py")

        self.assertIn("import senteval", source)
        self.assertNotIn("from sentence_transformers import SentenceTransformer", source)
        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)
        self.assertNotIn("self.control.should_save = True", source)

    def test_ddp_similarity_mask_expands_to_global_batch(self):
        source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = torch.ones", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = local_mask", source)
        self.assertIn("similarity_mask = similarity_mask.to(device=cos_sim.device", source)


if __name__ == "__main__":
    unittest.main()
