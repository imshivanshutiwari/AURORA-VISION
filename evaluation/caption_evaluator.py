from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("aurora.evaluation.caption")



@dataclass
class EvalReport:
    scores: Dict[str, float]
    predictions: List[str]
    references: List[str]
    n_samples: int

    @property
    def aggregate(self) -> float:
        return self.scores.get("aggregate", 0.0)


class MultiModalEvaluator:
    """
    Evaluates video captions using 4 metrics:
    BLEU-4, ROUGE-L, METEOR, BERTScore
    """

    def compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU-4 score using sacrebleu."""
        import sacrebleu

        if not predictions or not references:
            return 0.0

        result = sacrebleu.corpus_bleu(predictions, [references])
        return result.score / 100.0

    def compute_rouge(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        agg: Dict[str, Dict[str, List[float]]] = {
            m: {"precision": [], "recall": [], "fmeasure": []}
            for m in ["rouge1", "rouge2", "rougeL"]
        }

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for metric, score in scores.items():
                agg[metric]["precision"].append(score.precision)
                agg[metric]["recall"].append(score.recall)
                agg[metric]["fmeasure"].append(score.fmeasure)

        result = {}
        for metric, vals in agg.items():
            import numpy as np

            result[metric] = {
                "precision": float(np.mean(vals["precision"])),
                "recall": float(np.mean(vals["recall"])),
                "f1": float(np.mean(vals["fmeasure"])),
            }

        return result

    def compute_meteor(self, predictions: List[str], references: List[str]) -> float:
        """Compute METEOR score using a pure-Python unigram implementation.

        Implements the standard METEOR formula (no external dependencies):
          P, R  = unigram precision / recall over exact token matches
          F     = P*R / (alpha*P + (1-alpha)*R)   alpha=0.9
          penalty = gamma * (chunks/matches)^beta  gamma=0.5, beta=3
          METEOR  = F * (1 - penalty)
        """
        if not predictions or not references:
            return 0.0
        scores = [self._meteor_single(p, r) for p, r in zip(predictions, references)]
        return float(sum(scores) / max(len(scores), 1))

    def _meteor_single(self, prediction: str, reference: str) -> float:
        """METEOR score for one (prediction, reference) pair."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0

        # Build reference token counts
        ref_counts: Dict[str, int] = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1

        # Greedily match prediction tokens against remaining reference counts
        remaining = dict(ref_counts)
        matched_flags = []
        for token in pred_tokens:
            if remaining.get(token, 0) > 0:
                matched_flags.append(True)
                remaining[token] -= 1
            else:
                matched_flags.append(False)

        matches = sum(matched_flags)
        if matches == 0:
            return 0.0

        precision = matches / len(pred_tokens)
        recall = matches / len(ref_tokens)

        alpha = 0.9
        f_mean = precision * recall / (alpha * precision + (1 - alpha) * recall)

        # Count contiguous matched chunks in prediction order
        chunks = sum(
            1
            for i, v in enumerate(matched_flags)
            if v and (i == 0 or not matched_flags[i - 1])
        )
        gamma, beta = 0.5, 3
        penalty = gamma * ((chunks / matches) ** beta)
        return f_mean * (1.0 - penalty)

    def compute_bertscore(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute BERTScore using DeBERTa model."""
        try:
            from bert_score import score as bert_score_fn

            P, R, F1 = bert_score_fn(
                predictions,
                references,
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli",
                verbose=False,
                rescale_with_baseline=True,
            )
            return {
                "precision": float(P.mean().item()),
                "recall": float(R.mean().item()),
                "f1": float(F1.mean().item()),
            }
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}. Using simplified fallback.")
            return self._bertscore_fallback(predictions, references)

    def _bertscore_fallback(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Fallback using lighter BERTScore model."""
        try:
            from bert_score import score as bert_score_fn

            P, R, F1 = bert_score_fn(
                predictions,
                references,
                lang="en",
                model_type="bert-base-uncased",
                verbose=False,
            )
            return {
                "precision": float(P.mean().item()),
                "recall": float(R.mean().item()),
                "f1": float(F1.mean().item()),
            }
        except Exception as e:
            logger.warning(f"BERTScore fallback also failed: {e}")
            overlap_scores = [
                len(set(p.lower().split()) & set(r.lower().split()))
                / max(len(set(r.lower().split())), 1)
                for p, r in zip(predictions, references)
            ]
            import numpy as np

            f1 = float(np.mean(overlap_scores))
            return {"precision": f1, "recall": f1, "f1": f1}

    def full_evaluation(
        self, predictions: List[str], references: List[str]
    ) -> EvalReport:
        """Compute all 4 metrics and return an EvalReport."""
        if not predictions or not references:
            return EvalReport(
                scores={"bleu": 0.0, "rouge": 0.0, "meteor": 0.0, "bertscore": 0.0, "aggregate": 0.0},
                predictions=predictions,
                references=references,
                n_samples=0,
            )

        bleu = self.compute_bleu(predictions, references)
        rouge = self.compute_rouge(predictions, references)
        meteor = self.compute_meteor(predictions, references)
        bertscore = self.compute_bertscore(predictions, references)

        rouge_l = rouge.get("rougeL", {}).get("f1", 0.0)
        bs_f1 = bertscore.get("f1", 0.0)
        aggregate = float((bleu + rouge_l + meteor + bs_f1) / 4.0)

        scores = {
            "bleu": bleu,
            "rouge": rouge_l,
            "rouge_detail": rouge,
            "meteor": meteor,
            "bertscore": bs_f1,
            "bertscore_detail": bertscore,
            "aggregate": aggregate,
        }

        logger.info(
            f"Evaluation: BLEU={bleu:.3f} ROUGE-L={rouge_l:.3f} "
            f"METEOR={meteor:.3f} BERTScore={bs_f1:.3f} agg={aggregate:.3f}"
        )

        return EvalReport(
            scores=scores,
            predictions=predictions,
            references=references,
            n_samples=len(predictions),
        )
