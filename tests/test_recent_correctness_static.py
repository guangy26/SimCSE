import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class RecentCorrectnessStaticTests(unittest.TestCase):
    def source(self, relative_path):
        return (REPO_ROOT / relative_path).read_text()

    def test_train_main_initializes_results_before_return(self):
        tree = ast.parse(self.source("train.py"))
        main = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        result_assign_line = None
        return_line = None
        for node in ast.walk(main):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets)
            ):
                result_assign_line = node.lineno
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "results":
                return_line = node.lineno

        self.assertIsNotNone(result_assign_line)
        self.assertIsNotNone(return_line)
        self.assertLess(result_assign_line, return_line)

    def test_helper_similarity_mask_is_positive_and_preserves_diagonal(self):
        source = self.source("train.py")

        self.assertIn("with torch.no_grad():", source)
        self.assertIn("help_model.eval()", source)
        self.assertIn("F.cosine_similarity(", source)
        self.assertIn("similarity_mask = torch.ones_like(similarity_scores)", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertIn('batch["similarity_mask"] = similarity_mask', source)
        self.assertNotIn("batch[\"similarity_mask\"] = similarity_scores", source)

    def test_similarity_mask_expands_for_distributed_gather(self):
        source = self.source("simcse/models.py")

        self.assertIn("similarity_mask_list = [torch.ones_like(similarity_mask)", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = torch.ones_like(cos_sim)", source)
        self.assertIn("global_similarity_mask[start:end, start:end] = rank_similarity_mask", source)
        self.assertIn("similarity_mask.clamp_min(torch.finfo(cos_sim.dtype).tiny)", source)


if __name__ == "__main__":
    unittest.main()
