from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from evaluation.caption_evaluator import MultiModalEvaluator
from utils.logger import get_logger

logger = get_logger("aurora.evaluation.qa")


@dataclass
class QAEvalResult:
    exact_match: float
    f1_score: float
    bleu: float
    bertscore: float
    n_samples: int


class QAEvaluator:
    """
    Question-answering evaluation: Exact Match, F1, BLEU, BERTScore.
    Follows SQuAD-style evaluation for open-ended video QA.
    """

    def __init__(self):
        self.caption_evaluator = MultiModalEvaluator()

    def exact_match(self, prediction: str, reference: str) -> float:
        """Binary exact match after normalization."""
        return float(self._normalize(prediction) == self._normalize(reference))

    def token_f1(self, prediction: str, reference: str) -> float:
        """Token-level F1 (SQuAD-style)."""
        pred_tokens = set(self._normalize(prediction).split())
        ref_tokens = set(self._normalize(reference).split())

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = pred_tokens & ref_tokens
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)

    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
    ) -> QAEvalResult:
        """Evaluate a batch of QA pairs with all metrics."""
        if not predictions or not references:
            return QAEvalResult(exact_match=0.0, f1_score=0.0, bleu=0.0, bertscore=0.0, n_samples=0)

        em_scores = [self.exact_match(p, r) for p, r in zip(predictions, references)]
        f1_scores = [self.token_f1(p, r) for p, r in zip(predictions, references)]

        bleu = self.caption_evaluator.compute_bleu(predictions, references)
        bertscore = self.caption_evaluator.compute_bertscore(predictions, references)

        return QAEvalResult(
            exact_match=float(np.mean(em_scores)),
            f1_score=float(np.mean(f1_scores)),
            bleu=bleu,
            bertscore=bertscore.get("f1", 0.0),
            n_samples=len(predictions),
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        import re

        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text
