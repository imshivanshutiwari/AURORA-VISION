"""Full benchmark pipeline for AURORA-VISION evaluation."""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.caption_evaluator import EvalReport, MultiModalEvaluator
from evaluation.qa_evaluator import QAEvaluator
from evaluation.retrieval_evaluator import RetrievalEvaluator
from utils.logger import get_logger

logger = get_logger("aurora.evaluation.benchmark")


class BenchmarkRunner:
    """
    Runs the complete AURORA-VISION evaluation suite:
    - Caption evaluation (BLEU/ROUGE/METEOR/BERTScore)
    - Retrieval (R@1/R@5/R@10)
    - QA accuracy + F1
    """

    def __init__(self, output_dir: str = "assets/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.caption_evaluator = MultiModalEvaluator()
        self.qa_evaluator = QAEvaluator()
        self.retrieval_evaluator = RetrievalEvaluator()

    def run_caption_benchmark(
        self,
        predictions: List[str],
        references: List[str],
        save_results: bool = True,
    ) -> EvalReport:
        """Run full caption evaluation and save results."""
        logger.info(f"Running caption benchmark on {len(predictions)} samples")
        report = self.caption_evaluator.full_evaluation(predictions, references)

        if save_results:
            out_path = self.output_dir / f"caption_benchmark_{int(time.time())}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "scores": {k: v for k, v in report.scores.items() if isinstance(v, float)},
                        "n_samples": report.n_samples,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Caption benchmark results saved: {out_path}")

        return report

    def run_retrieval_benchmark(
        self,
        text_embeddings,
        video_embeddings,
        ground_truth: Optional[List[int]] = None,
    ) -> Dict:
        """Run retrieval benchmark in both directions."""
        logger.info("Running retrieval benchmark")
        t2v = self.retrieval_evaluator.evaluate_text_to_video(
            text_embeddings, video_embeddings, ground_truth
        )
        v2t = self.retrieval_evaluator.evaluate_video_to_text(
            video_embeddings, text_embeddings, ground_truth
        )

        results = {
            "text_to_video": {
                "R@1": t2v.r_at_1,
                "R@5": t2v.r_at_5,
                "R@10": t2v.r_at_10,
                "median_rank": t2v.median_rank,
            },
            "video_to_text": {
                "R@1": v2t.r_at_1,
                "R@5": v2t.r_at_5,
                "R@10": v2t.r_at_10,
                "median_rank": v2t.median_rank,
            },
        }
        logger.info(f"Retrieval results: {results}")
        return results

    def run_qa_benchmark(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict:
        """Run QA evaluation."""
        logger.info(f"Running QA benchmark on {len(predictions)} samples")
        result = self.qa_evaluator.evaluate_batch(predictions, references)
        return {
            "exact_match": result.exact_match,
            "f1": result.f1_score,
            "bleu": result.bleu,
            "bertscore": result.bertscore,
            "n_samples": result.n_samples,
        }

    def run_full_benchmark(
        self,
        caption_preds: List[str],
        caption_refs: List[str],
        qa_preds: Optional[List[str]] = None,
        qa_refs: Optional[List[str]] = None,
    ) -> Dict:
        """Run all evaluation components and return combined results."""
        results = {}

        results["caption"] = self.run_caption_benchmark(
            caption_preds, caption_refs
        ).scores

        if qa_preds and qa_refs:
            results["qa"] = self.run_qa_benchmark(qa_preds, qa_refs)

        out_path = self.output_dir / f"full_benchmark_{int(time.time())}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Full benchmark complete. Results: {out_path}")
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    runner = BenchmarkRunner()
    preds = ["A person cooking in a kitchen."] * args.n_samples
    refs = ["Someone is preparing food in the kitchen."] * args.n_samples

    results = runner.run_full_benchmark(preds, refs)
    logger.info(f"Benchmark complete: {results}")


if __name__ == "__main__":
    main()
