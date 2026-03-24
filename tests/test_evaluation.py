"""Tests for evaluation/ modules: MultiModalEvaluator (BLEU, ROUGE, METEOR, BERTScore)."""
import unittest


class TestMultiModalEvaluatorBLEU(unittest.TestCase):
    """Tests for BLEU computation."""

    def setUp(self):
        from evaluation.caption_evaluator import MultiModalEvaluator

        self.evaluator = MultiModalEvaluator()

    def test_bleu_perfect_prediction(self):
        """BLEU score is 1.0 when prediction equals reference."""
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        score = self.evaluator.compute_bleu(preds, refs)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_bleu_empty_inputs_returns_zero(self):
        """BLEU returns 0.0 for empty prediction list."""
        score = self.evaluator.compute_bleu([], [])
        self.assertAlmostEqual(score, 0.0)

    def test_bleu_score_range(self):
        """BLEU score is in [0, 1] range."""
        preds = ["a dog runs in the field"]
        refs = ["the cat sat on the mat"]
        score = self.evaluator.compute_bleu(preds, refs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestMultiModalEvaluatorROUGE(unittest.TestCase):
    """Tests for ROUGE computation."""

    def setUp(self):
        from evaluation.caption_evaluator import MultiModalEvaluator

        self.evaluator = MultiModalEvaluator()

    def test_rouge_returns_all_metrics(self):
        """compute_rouge returns rouge1, rouge2, rougeL keys."""
        preds = ["the quick brown fox jumps over the lazy dog"]
        refs = ["the quick brown fox leaps over the lazy cat"]
        result = self.evaluator.compute_rouge(preds, refs)
        self.assertIn("rouge1", result)
        self.assertIn("rouge2", result)
        self.assertIn("rougeL", result)

    def test_rouge_perfect_match_f1(self):
        """ROUGE F1 is ~1.0 for identical prediction and reference."""
        text = "a person walks down the street"
        result = self.evaluator.compute_rouge([text], [text])
        self.assertAlmostEqual(result["rougeL"]["f1"], 1.0, places=3)


class TestMultiModalEvaluatorMETEOR(unittest.TestCase):
    """Tests for METEOR computation."""

    def setUp(self):
        from evaluation.caption_evaluator import MultiModalEvaluator

        self.evaluator = MultiModalEvaluator()

    def test_meteor_score_range(self):
        """METEOR score is in [0, 1] range."""
        preds = ["the dog runs fast"]
        refs = ["a dog is running quickly"]
        score = self.evaluator.compute_meteor(preds, refs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_meteor_perfect_match(self):
        """METEOR score is >= 0.99 for identical prediction and reference."""
        text = "the video shows a person cooking"
        score = self.evaluator.compute_meteor([text], [text])
        self.assertGreaterEqual(score, 0.99)


class TestMultiModalEvaluatorFullEval(unittest.TestCase):
    """Tests for full_evaluation aggregation."""

    def setUp(self):
        from evaluation.caption_evaluator import MultiModalEvaluator

        self.evaluator = MultiModalEvaluator()

    def test_full_evaluation_aggregate_in_range(self):
        """full_evaluation aggregate score is in [0, 1]."""
        preds = ["a person is cooking food in a kitchen"]
        refs = ["someone prepares a meal in the kitchen"]
        report = self.evaluator.full_evaluation(preds, refs)
        self.assertGreaterEqual(report.aggregate, 0.0)
        self.assertLessEqual(report.aggregate, 1.0)

    def test_full_evaluation_empty_returns_zero_aggregate(self):
        """full_evaluation on empty inputs returns aggregate=0.0 and n_samples=0."""
        report = self.evaluator.full_evaluation([], [])
        self.assertEqual(report.n_samples, 0)
        self.assertAlmostEqual(report.aggregate, 0.0)


if __name__ == "__main__":
    unittest.main()
