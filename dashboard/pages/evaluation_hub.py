"""
Page 4 — Evaluation Hub (VIZ18-VIZ20)
"""
from typing import Any, Dict, List, Optional

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from dashboard.theme import (
    ACCENT_TEAL,
    BG_CARD,
    BG_PANEL,
    BORDER,
    CARD_STYLE,
    CRITICAL,
    FONT,
    INFO,
    PLOTLY_THEME,
    SUCCESS,
    TEXT,
    TEXT_DIM,
    WARNING,
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_THEME["layout"])
    return fig


def viz18_metrics_over_time(
    metric_history: Dict[str, List[float]] = None,
    n_videos: int = 20,
) -> dbc.Card:
    """VIZ18: BLEU/ROUGE/METEOR/BERTScore over time with threshold lines."""
    if metric_history is None:
        np.random.seed(42)
        metric_history = {
            "BLEU": np.clip(np.random.normal(0.45, 0.1, n_videos), 0, 1).tolist(),
            "ROUGE-L": np.clip(np.random.normal(0.52, 0.09, n_videos), 0, 1).tolist(),
            "METEOR": np.clip(np.random.normal(0.48, 0.11, n_videos), 0, 1).tolist(),
            "BERTScore": np.clip(np.random.normal(0.61, 0.07, n_videos), 0, 1).tolist(),
        }

    colors = [ACCENT_TEAL, INFO, WARNING, SUCCESS]
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"<span style='color:{TEXT};font-size:9px'>{m}</span>"
            for m in metric_history.keys()
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (metric, vals), (row, col), color in zip(metric_history.items(), positions, colors):
        x = list(range(1, len(vals) + 1))
        rolling_mean = np.convolve(vals, np.ones(5) / 5, mode="same").tolist()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=vals,
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.6),
                name=f"{metric} per-video",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rolling_mean,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{metric} avg",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_hline(
            y=0.6,
            line_dash="dot",
            line_color=SUCCESS,
            line_width=1,
            row=row,
            col=col,
        )
        fig.add_hline(
            y=0.4,
            line_dash="dot",
            line_color=WARNING,
            line_width=1,
            row=row,
            col=col,
        )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Evaluation Metrics Over Videos", font=dict(size=11, color=TEXT)),
        height=380,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "📈 VIZ18 — METRICS OVER TIME",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="metrics-over-time", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def viz19_caption_quality_scatter(
    video_scores: List[Dict] = None,
) -> dbc.Card:
    """VIZ19: BLEU vs BERTScore scatter colored by video category."""
    if video_scores is None:
        np.random.seed(42)
        categories = ["action", "lecture", "news"]
        video_scores = []
        for cat in categories:
            for _ in range(15):
                video_scores.append(
                    {
                        "bleu": float(np.clip(np.random.normal(0.45, 0.15), 0, 1)),
                        "bertscore": float(np.clip(np.random.normal(0.58, 0.10), 0, 1)),
                        "category": cat,
                        "caption": f"A {cat} video showing various activities.",
                    }
                )

    categories_unique = list(set(v["category"] for v in video_scores))
    colors = [ACCENT_TEAL, INFO, WARNING, SUCCESS, CRITICAL]

    fig = go.Figure()
    for i, cat in enumerate(categories_unique):
        cat_data = [v for v in video_scores if v["category"] == cat]
        fig.add_trace(
            go.Scatter(
                x=[v["bleu"] for v in cat_data],
                y=[v["bertscore"] for v in cat_data],
                mode="markers",
                name=cat,
                marker=dict(
                    color=colors[i % len(colors)],
                    size=8,
                    opacity=0.75,
                ),
                hovertemplate=f"<b>{cat}</b><br>BLEU: %{{x:.3f}}<br>BERTScore: %{{y:.3f}}<br>%{{text}}<extra></extra>",
                text=[v.get("caption", "")[:50] for v in cat_data],
            )
        )

    fig.add_shape(type="line", x0=0.6, x1=0.6, y0=0, y1=1, line=dict(color=SUCCESS, dash="dot", width=1))
    fig.add_shape(type="line", x0=0, x1=1, y0=0.6, y1=0.6, line=dict(color=SUCCESS, dash="dot", width=1))

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Caption Quality: BLEU vs BERTScore (by category)", font=dict(size=11, color=TEXT)),
        xaxis_title="BLEU-4",
        yaxis_title="BERTScore F1",
        height=320,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "◎ VIZ19 — CAPTION QUALITY SCATTER",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="caption-scatter", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def viz20_retrieval_benchmark(
    retrieval_metrics: Dict[str, Dict[str, float]] = None
) -> dbc.Card:
    """VIZ20: R@1/R@5/R@10 retrieval benchmark (MSR-VTT)."""
    if retrieval_metrics is None:
        retrieval_metrics = {
            "Text→Video": {"R@1": 0.312, "R@5": 0.578, "R@10": 0.682},
            "Video→Text": {"R@1": 0.335, "R@5": 0.601, "R@10": 0.714},
        }

    fig = go.Figure()
    k_labels = ["R@1", "R@5", "R@10"]
    colors = [ACCENT_TEAL, INFO]

    for (direction, metrics), color in zip(retrieval_metrics.items(), colors):
        fig.add_trace(
            go.Bar(
                x=k_labels,
                y=[metrics.get(k, 0) for k in k_labels],
                name=direction,
                marker_color=color,
                text=[f"{metrics.get(k, 0):.3f}" for k in k_labels],
                textposition="outside",
                textfont=dict(color=color, size=9),
            )
        )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="MSR-VTT Retrieval Benchmark", font=dict(size=11, color=TEXT)),
        yaxis=dict(range=[0, 1]),
        barmode="group",
        height=280,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "⟲ VIZ20 — RETRIEVAL BENCHMARK",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="retrieval-benchmark", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def build_evaluation_hub_layout() -> html.Div:
    """Assemble the Evaluation Hub page."""
    return html.Div(
        [
            dbc.Row([dbc.Col(viz18_metrics_over_time(), width=12)], className="mb-2"),
            dbc.Row(
                [
                    dbc.Col(viz19_caption_quality_scatter(), width=7),
                    dbc.Col(viz20_retrieval_benchmark(), width=5),
                ],
                className="mb-2",
            ),
        ]
    )
