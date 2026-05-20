import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_train_script_no_longer_requires_sentence_transformers(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            tree = ast.parse(read_source(relative_path))
            imports = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            self.assertFalse(
                any(
                    (isinstance(node, ast.Import) and any(alias.name == "sentence_transformers" for alias in node.names))
                    or (isinstance(node, ast.ImportFrom) and node.module == "sentence_transformers")
                    for node in imports
                ),
                f"{relative_path} should not import the unused sentence_transformers dependency",
            )

    def test_train_main_returns_initialized_results(self):
        tree = ast.parse(read_source("train.py"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        saw_results_assignment = False
        for node in ast.walk(main):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == "results" for target in targets):
                    saw_results_assignment = True
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results":
                self.assertTrue(saw_results_assignment, "main() returns results without initializing it")
                break
        else:
            self.fail("main() should return results")

    def test_mlm_masking_path_is_implemented(self):
        source = read_source("train.py")
        self.assertIn("def mask_tokens(", source)
        self.assertIn("inputs = inputs.clone()", source)
        self.assertIn("labels[~masked_indices] = -100", source)
        self.assertIn("return inputs, labels", source)
        self.assertNotIn("def mask_tokens(\n            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None\n        ) -> Tuple[torch.Tensor, torch.Tensor]:\n            \"\"\"\n            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.\n            \"\"\"\n            pass", source)

    def test_helper_similarity_mask_is_frozen_raw_cosine_and_keeps_diagonal(self):
        source = read_source("train.py")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.nn.functional.cosine_similarity", source)
        self.assertIn("similarity_mask[diag, diag] = 1.0", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("parameter.requires_grad_(False)", source)

    def test_distributed_similarity_mask_matches_gathered_logits(self):
        source = read_source("simcse/models.py")
        self.assertIn("def gather_distributed_similarity_mask(similarity_mask):", source)
        self.assertIn("dist.all_gather(tensor_list=mask_list", source)
        self.assertIn("global_mask = similarity_mask.new_ones", source)
        self.assertIn("similarity_mask = gather_distributed_similarity_mask(similarity_mask)", source)
        self.assertIn("similarity_mask = similarity_mask.to(device=cos_sim.device, dtype=cos_sim.dtype)", source)

    def test_trainer_evaluation_imports_senteval(self):
        source = read_source("simcse/trainers.py")
        self.assertIn("import senteval", source)
        self.assertIn("se = senteval.engine.SE(params, batcher, prepare)", source)


if __name__ == "__main__":
    unittest.main()
