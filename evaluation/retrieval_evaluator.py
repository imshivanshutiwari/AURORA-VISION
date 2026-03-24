from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger("aurora.evaluation.retrieval")


@dataclass
class RetrievalMetrics:
    r_at_1: float
    r_at_5: float
    r_at_10: float
    median_rank: float
    mean_rank: float


class RetrievalEvaluator:
    """
    Video-text retrieval evaluation: R@1, R@5, R@10 on MSR-VTT benchmark.
    Evaluates both text-to-video and video-to-text retrieval directions.
    """

    def compute_recall_at_k(
        self,
        similarity_matrix: np.ndarray,
        ground_truth: np.ndarray,
        k_values: List[int] = None,
    ) -> Dict[int, float]:
        """
        Compute recall@K for a similarity matrix.

        Args:
            similarity_matrix: (N_queries, N_candidates) float array
            ground_truth: (N_queries,) int array with correct candidate indices

        Returns:
            Dict[k, recall@k]
        """
        if k_values is None:
            k_values = [1, 5, 10]

        n_queries = similarity_matrix.shape[0]
        results = {}

        for k in k_values:
            correct = 0
            for i in range(n_queries):
                top_k = np.argsort(-similarity_matrix[i])[:k]
                if ground_truth[i] in top_k:
                    correct += 1
            results[k] = correct / n_queries

        return results

    def evaluate_text_to_video(
        self,
        text_embeddings: torch.Tensor,
        video_embeddings: torch.Tensor,
        ground_truth_indices: Optional[List[int]] = None,
    ) -> RetrievalMetrics:
        """
        Text-to-video retrieval: for each text query, rank all videos.
        """
        text_norm = F.normalize(text_embeddings.float(), dim=-1)
        video_norm = F.normalize(video_embeddings.float(), dim=-1)
        sim_matrix = torch.matmul(text_norm, video_norm.T).cpu().numpy()

        n = sim_matrix.shape[0]
        if ground_truth_indices is None:
            ground_truth_indices = list(range(n))

        gt = np.array(ground_truth_indices)
        recall = self.compute_recall_at_k(sim_matrix, gt, k_values=[1, 5, 10])

        ranks = []
        for i in range(n):
            sorted_idx = np.argsort(-sim_matrix[i])
            rank = int(np.where(sorted_idx == gt[i])[0][0]) + 1
            ranks.append(rank)

        return RetrievalMetrics(
            r_at_1=recall[1],
            r_at_5=recall[5],
            r_at_10=recall[10],
            median_rank=float(np.median(ranks)),
            mean_rank=float(np.mean(ranks)),
        )

    def evaluate_video_to_text(
        self,
        video_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        ground_truth_indices: Optional[List[int]] = None,
    ) -> RetrievalMetrics:
        """
        Video-to-text retrieval: for each video, rank all text captions.
        """
        return self.evaluate_text_to_video(
            text_embeddings=video_embeddings,
            video_embeddings=text_embeddings,
            ground_truth_indices=ground_truth_indices,
        )

    def run_msrvtt_benchmark(
        self,
        clip_encoder,
        msrvtt_fetcher,
        n_videos: int = 100,
    ) -> Dict[str, RetrievalMetrics]:
        """Run full MSR-VTT text-video retrieval benchmark."""
        logger.info(f"Running MSR-VTT benchmark on {n_videos} videos")

        benchmark_df = msrvtt_fetcher.build_retrieval_benchmark()
        if benchmark_df.empty:
            logger.warning("No benchmark data available")
            return {}

        video_ids = benchmark_df["video_id"].unique()[:n_videos].tolist()
        video_embeddings = []
        text_embeddings = []

        for vid_id in video_ids:
            descriptions = msrvtt_fetcher.get_descriptions(vid_id)
            if not descriptions:
                continue
            text_emb = clip_encoder.encode_text([descriptions[0]])
            text_embeddings.append(text_emb)

            video_path = msrvtt_fetcher.fetch_video(vid_id)
            if video_path:
                from data.processors.frame_sampler import AdaptiveFrameSampler

                sampler = AdaptiveFrameSampler()
                frames = sampler.sample_uniform(video_path, n_frames=8)
                if frames:
                    vid_emb = clip_encoder.encode_frames(frames).mean(dim=0, keepdim=True)
                    video_embeddings.append(vid_emb)
                else:
                    import torch

                    video_embeddings.append(torch.zeros(1, 768))
            else:
                import torch

                video_embeddings.append(torch.zeros(1, 768))

        if not video_embeddings:
            return {}

        import torch

        vid_embs = torch.cat(video_embeddings, dim=0)
        txt_embs = torch.cat(text_embeddings, dim=0)
        gt = list(range(len(video_embeddings)))

        return {
            "text_to_video": self.evaluate_text_to_video(txt_embs, vid_embs, gt),
            "video_to_text": self.evaluate_video_to_text(vid_embs, txt_embs, gt),
        }
