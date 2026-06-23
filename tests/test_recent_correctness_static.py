import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class RecentCorrectnessStaticTests(unittest.TestCase):
    def test_training_modules_do_not_import_undeclared_sentence_transformers(self):
        for relative_path in ("train.py", "simcse/trainers.py"):
            with self.subTest(relative_path=relative_path):
                tree = ast.parse(read_source(relative_path))
                imported_modules = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_modules.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module is not None:
                        imported_modules.append(node.module)

                self.assertNotIn("sentence_transformers", imported_modules)

    def test_train_results_is_initialized_before_return(self):
        source = read_source("train.py")

        self.assertIn("results = {}", source)
        self.assertLess(source.index("results = {}"), source.rindex("return results"))

    def test_mlm_mask_tokens_is_implemented_and_non_mutating(self):
        source = read_source("train.py")
        mask_tokens_start = source.index("def mask_tokens(")
        mask_tokens_source = source[mask_tokens_start:source.index("if model_args.help_model_path", mask_tokens_start)]

        self.assertNotIn("\n            pass\n", mask_tokens_source)
        self.assertIn("labels = inputs.clone()", mask_tokens_source)
        self.assertIn("inputs = inputs.clone()", mask_tokens_source)
        self.assertIn("return inputs, labels", mask_tokens_source)

    def test_helper_similarity_mask_uses_raw_cosine_and_preserves_positives(self):
        source = read_source("train.py")

        self.assertIn("with torch.no_grad():", source)
        self.assertIn("torch.nn.functional.cosine_similarity", source)
        self.assertIn("similarity_mask.fill_diagonal_(1.0)", source)
        self.assertIn("parameter.requires_grad_(False)", source)
        self.assertNotIn("Similarity(0.05)", source)

    def test_senteval_is_imported_after_path_setup(self):
        source = read_source("simcse/trainers.py")

        path_setup_index = source.index("sys.path.insert(0, PATH_TO_SENTEVAL)")
        import_index = source.index("import senteval")
        self.assertLess(path_setup_index, import_index)

    def test_trainer_does_not_force_disable_callback_evaluation_or_saving(self):
        source = read_source("simcse/trainers.py")

        self.assertNotIn("self.control.should_evaluate = False", source)
        self.assertNotIn("self.control.should_save = False", source)
        self.assertNotIn("self.control.should_save = True", source)

    def test_distributed_similarity_mask_expands_to_global_logits_shape(self):
        source = read_source("simcse/models.py")

        self.assertIn("similarity_mask_list = [torch.ones_like(similarity_mask) for _ in range(world_size)]", source)
        self.assertIn("dist.all_gather(tensor_list=similarity_mask_list", source)
        self.assertIn("global_similarity_mask = similarity_mask.new_ones", source)
        self.assertIn("global_similarity_mask[rank_start:rank_end, rank_start:rank_end] = rank_similarity_mask", source)
        self.assertIn("torch.log(similarity_mask.to(cos_sim.device))", source)


if __name__ == "__main__":
    unittest.main()
