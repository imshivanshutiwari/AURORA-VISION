"""
Page 1 — Vision Ops Center (VIZ01-VIZ07)
"""
from typing import Any, Dict, List

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


def viz01_video_player(video_src: str = "", transcription_segments: List[Dict] = None) -> dbc.Card:
    """VIZ01: Video player + transcription + OCR overlay + CLIP timeline."""
    segs = transcription_segments or []
    transcript_items = []
    for seg in segs[:20]:
        speaker = seg.get("speaker", "SPK")
        color = ACCENT_TEAL if seg.get("is_primary") else INFO
        transcript_items.append(
            html.Div(
                [
                    html.Span(
                        f"[{seg.get('start', 0):.1f}s {speaker}] ",
                        style={"color": color, "fontWeight": "bold"},
                    ),
                    html.Span(seg.get("text", ""), style={"color": TEXT}),
                ],
                style={"marginBottom": "4px", "fontSize": "11px"},
            )
        )

    return dbc.Card(
        [
            dbc.CardHeader(
                html.Span("▶ VIZ01 — VIDEO PLAYER + ANALYSIS", style={"color": ACCENT_TEAL}),
                style={"backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Input(
                                        id="video-url-input",
                                        type="text",
                                        placeholder="Paste YouTube URL or file path...",
                                        style={
                                            "width": "100%",
                                            "backgroundColor": BG_PANEL,
                                            "color": TEXT,
                                            "border": f"1px solid {BORDER}",
                                            "fontFamily": FONT,
                                            "padding": "6px",
                                            "marginBottom": "8px",
                                        },
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        id="video-query-input",
                                        type="text",
                                        placeholder="Enter analysis query...",
                                        value="Describe what is happening in this video.",
                                        style={
                                            "width": "100%",
                                            "backgroundColor": BG_PANEL,
                                            "color": TEXT,
                                            "border": f"1px solid {BORDER}",
                                            "fontFamily": FONT,
                                            "padding": "6px",
                                            "marginBottom": "8px",
                                        },
                                        debounce=True,
                                    ),
                                    html.Button(
                                        "▶ ANALYZE VIDEO",
                                        id="analyze-btn",
                                        style={
                                            "backgroundColor": ACCENT_TEAL,
                                            "color": BG_PANEL,
                                            "border": "none",
                                            "padding": "8px 20px",
                                            "fontFamily": FONT,
                                            "cursor": "pointer",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    html.Div(id="video-player-container", style={"marginTop": "12px"}),
                                ],
                                width=7,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        "TRANSCRIPT + SPEAKER LABELS",
                                        style={
                                            "color": TEXT_DIM,
                                            "fontSize": "10px",
                                            "marginBottom": "6px",
                                            "letterSpacing": "2px",
                                        },
                                    ),
                                    html.Div(
                                        transcript_items or [
                                            html.Span("No transcription yet.", style={"color": TEXT_DIM})
                                        ],
                                        id="transcript-panel",
                                        style={
                                            "height": "250px",
                                            "overflowY": "auto",
                                            "backgroundColor": BG_PANEL,
                                            "padding": "8px",
                                            "border": f"1px solid {BORDER}",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="clip-similarity-timeline",
                                        figure=_make_clip_timeline(),
                                        config={"displayModeBar": False},
                                        style={"height": "80px"},
                                    ),
                                ],
                                width=5,
                            ),
                        ]
                    )
                ]
            ),
        ],
        style=CARD_STYLE,
    )


def _make_clip_timeline(scores: List[float] = None) -> go.Figure:
    scores = scores or [0.0] * 32
    fig = go.Figure(
        go.Bar(
            x=list(range(len(scores))),
            y=scores,
            marker=dict(
                color=scores,
                colorscale=[[0, TEXT_DIM], [1, ACCENT_TEAL]],
                cmin=0,
                cmax=1,
            ),
            name="CLIP similarity",
        )
    )
    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="CLIP Frame Similarity", font=dict(size=10, color=TEXT)),
        xaxis_title="Frame",
        yaxis_title="Sim",
        showlegend=False,
        height=80,
        margin=dict(l=30, r=10, t=25, b=20),
    )
    return fig


def viz02_pipeline_waterfall(stage_latencies: Dict[str, float] = None) -> dbc.Card:
    """VIZ02: Multi-modal pipeline waterfall chart."""
    stages = stage_latencies or {
        "Ingest": 1200,
        "Visual": 2400,
        "Audio": 1800,
        "OCR": 900,
        "Fusion": 300,
        "QA": 1500,
        "Eval": 200,
    }

    names = list(stages.keys())
    values = list(stages.values())
    colors = [ACCENT_TEAL, INFO, WARNING, SUCCESS, "#a855f7", CRITICAL, TEXT_DIM]

    fig = go.Figure()
    for i, (name, val) in enumerate(zip(names, values)):
        fig.add_trace(
            go.Bar(
                x=[val],
                y=[name],
                orientation="h",
                name=name,
                marker_color=colors[i % len(colors)],
                text=[f"{val}ms"],
                textposition="inside",
                insidetextanchor="middle",
            )
        )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Pipeline Stage Latency (ms)", font=dict(size=11, color=TEXT)),
        barmode="overlay",
        showlegend=False,
        height=220,
        xaxis_title="Latency (ms)",
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "⟳ VIZ02 — MULTI-MODAL PIPELINE WATERFALL",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(
                dcc.Graph(
                    id="pipeline-waterfall",
                    figure=fig,
                    config={"displayModeBar": False},
                )
            ),
        ],
        style=CARD_STYLE,
    )


def viz03_frame_gallery(frames_data: List[Dict] = None) -> dbc.Card:
    """VIZ03: Keyframe gallery with scene boundary indicators."""
    frames_data = frames_data or [
        {"frame_idx": i, "timestamp": i, "scene_id": i // 4, "is_boundary": i % 4 == 0, "clip_score": 0.5}
        for i in range(8)
    ]

    fig = go.Figure()
    for i, fd in enumerate(frames_data[:16]):
        color = ACCENT_TEAL if fd.get("is_boundary") else INFO
        opacity = 0.3 + 0.7 * fd.get("clip_score", 0.5)
        fig.add_shape(
            type="rect",
            x0=i - 0.4,
            x1=i + 0.4,
            y0=0,
            y1=1,
            line=dict(color=color, width=2),
            fillcolor=f"rgba(29, 158, 117, {opacity * 0.3})",
        )
        fig.add_annotation(
            x=i,
            y=0.5,
            text=f"F{fd.get('frame_idx', i)}<br>{fd.get('timestamp', i):.1f}s<br>sim:{fd.get('clip_score', 0.5):.2f}",
            showarrow=False,
            font=dict(color=TEXT, size=8, family=FONT),
        )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Keyframe Gallery — Click to Jump", font=dict(size=11, color=TEXT)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=180,
        clickmode="event+select",
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "⬛ VIZ03 — FRAME GALLERY",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(
                dcc.Graph(id="frame-gallery", figure=fig, config={"displayModeBar": False})
            ),
        ],
        style=CARD_STYLE,
    )


def viz04_modality_gates(gate_weights: Dict[str, float] = None) -> dbc.Card:
    """VIZ04: Modality gate weight gauges (Visual / Audio / Text)."""
    weights = gate_weights or {"visual": 0.55, "audio": 0.25, "text": 0.20}

    fig = go.Figure()
    positions = {"visual": 0, "audio": 1, "text": 2}
    colors = {"visual": ACCENT_TEAL, "audio": INFO, "text": WARNING}
    labels = {"visual": "VISUAL", "audio": "AUDIO", "text": "TEXT (OCR)"}

    for modality, idx in positions.items():
        val = weights.get(modality, 0.33)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=round(val * 100, 1),
                title={"text": labels[modality], "font": {"color": TEXT, "size": 10}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": TEXT, "tickfont": {"size": 8}},
                    "bar": {"color": colors[modality]},
                    "bgcolor": BG_PANEL,
                    "bordercolor": BORDER,
                    "steps": [
                        {"range": [0, 33], "color": BG_CARD},
                        {"range": [33, 66], "color": ACCENT_DIM},
                        {"range": [66, 100], "color": BORDER},
                    ],
                },
                domain={"row": 0, "column": idx},
                number={"suffix": "%", "font": {"color": colors[modality], "size": 14}},
            )
        )

    _apply_theme(fig)
    fig.update_layout(
        grid={"rows": 1, "columns": 3, "pattern": "independent"},
        height=200,
        title=dict(text="Modality Gate Weights (per query)", font=dict(size=11, color=TEXT)),
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "⚖ VIZ04 — MODALITY GATE WEIGHTS",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(
                dcc.Graph(id="modality-gates", figure=fig, config={"displayModeBar": False})
            ),
        ],
        style=CARD_STYLE,
    )


def viz05_qa_interface() -> dbc.Card:
    """VIZ05: Q&A interface with multi-modal citations."""
    return dbc.Card(
        [
            dbc.CardHeader(
                "? VIZ05 — MULTI-MODAL Q&A",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(
                [
                    dcc.Textarea(
                        id="qa-answer-display",
                        value="Analysis results will appear here after processing a video...",
                        style={
                            "width": "100%",
                            "height": "120px",
                            "backgroundColor": BG_PANEL,
                            "color": TEXT,
                            "border": f"1px solid {BORDER}",
                            "fontFamily": FONT,
                            "fontSize": "11px",
                            "padding": "8px",
                            "resize": "none",
                        },
                        readOnly=True,
                    ),
                    html.Div(
                        id="citations-panel",
                        style={
                            "marginTop": "8px",
                            "padding": "6px",
                            "backgroundColor": BG_PANEL,
                            "border": f"1px solid {BORDER}",
                            "fontSize": "10px",
                            "color": TEXT_DIM,
                        },
                        children="Citations will appear here...",
                    ),
                ]
            ),
        ],
        style=CARD_STYLE,
    )


def viz06_processing_queue(queue_data: List[Dict] = None) -> dbc.Card:
    """VIZ06: Video processing queue table."""
    queue_data = queue_data or []

    rows = [
        html.Tr(
            [
                html.Th("URL", style={"color": TEXT_DIM}),
                html.Th("STATUS", style={"color": TEXT_DIM}),
                html.Th("PROGRESS", style={"color": TEXT_DIM}),
                html.Th("ETA", style={"color": TEXT_DIM}),
            ],
            style={"borderBottom": f"1px solid {BORDER}"},
        )
    ]

    status_colors = {
        "DONE": SUCCESS,
        "PROCESSING": ACCENT_TEAL,
        "DOWNLOADING": INFO,
        "QUEUED": WARNING,
        "FAILED": CRITICAL,
    }

    for item in queue_data[:10]:
        status = item.get("status", "QUEUED")
        color = status_colors.get(status, TEXT)
        rows.append(
            html.Tr(
                [
                    html.Td(
                        item.get("url", "")[:40] + "...",
                        style={"color": TEXT, "fontSize": "10px"},
                    ),
                    html.Td(status, style={"color": color, "fontWeight": "bold", "fontSize": "10px"}),
                    html.Td(
                        f"{item.get('progress', 0):.0f}%",
                        style={"color": TEXT, "fontSize": "10px"},
                    ),
                    html.Td(item.get("eta", "—"), style={"color": TEXT_DIM, "fontSize": "10px"}),
                ],
                style={"borderBottom": f"1px solid {BORDER}"},
            )
        )

    return dbc.Card(
        [
            dbc.CardHeader(
                "⬡ VIZ06 — VIDEO PROCESSING QUEUE",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(
                [
                    html.Table(
                        rows,
                        id="queue-table",
                        style={
                            "width": "100%",
                            "fontFamily": FONT,
                            "borderCollapse": "collapse",
                            "fontSize": "11px",
                        },
                    ),
                    html.Div(
                        id="queue-throughput",
                        style={"marginTop": "8px", "color": TEXT_DIM, "fontSize": "10px"},
                        children="Throughput: — videos/hour",
                    ),
                ]
            ),
        ],
        style=CARD_STYLE,
    )


def viz07_evaluation_scores(scores: Dict[str, float] = None) -> dbc.Card:
    """VIZ07: Evaluation metric badges (BLEU/ROUGE/METEOR/BERTScore)."""
    scores = scores or {"bleu": 0.0, "rouge": 0.0, "meteor": 0.0, "bertscore": 0.0}

    def _badge_color(v: float) -> str:
        if v >= 0.6:
            return SUCCESS
        elif v >= 0.4:
            return WARNING
        return CRITICAL

    badges = []
    metric_labels = {"bleu": "BLEU-4", "rouge": "ROUGE-L", "meteor": "METEOR", "bertscore": "BERTScore"}
    for key, label in metric_labels.items():
        val = scores.get(key, 0.0)
        color = _badge_color(val)
        badges.append(
            dbc.Col(
                html.Div(
                    [
                        html.Div(label, style={"fontSize": "9px", "color": TEXT_DIM, "letterSpacing": "1px"}),
                        html.Div(
                            f"{val:.3f}",
                            style={"fontSize": "22px", "color": color, "fontWeight": "bold"},
                        ),
                    ],
                    style={
                        "textAlign": "center",
                        "padding": "12px",
                        "backgroundColor": BG_PANEL,
                        "border": f"2px solid {color}",
                        "borderRadius": "4px",
                        "fontFamily": FONT,
                    },
                ),
                width=3,
            )
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(scores))),
            y=list(scores.values()),
            mode="lines+markers",
            line=dict(color=ACCENT_TEAL, width=2),
            marker=dict(color=[_badge_color(v) for v in scores.values()], size=8),
        )
    )
    _apply_theme(fig)
    fig.update_layout(height=100, showlegend=False, margin=dict(l=20, r=10, t=10, b=20))

    return dbc.Card(
        [
            dbc.CardHeader(
                "◎ VIZ07 — EVALUATION SCORES",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(
                [
                    dbc.Row(badges, className="mb-3"),
                    dcc.Graph(
                        id="eval-scores-trend",
                        figure=fig,
                        config={"displayModeBar": False},
                    ),
                ]
            ),
        ],
        style=CARD_STYLE,
    )


def build_vision_ops_layout() -> html.Div:
    """Assemble the complete Vision Ops Center page."""
    return html.Div(
        [
            dbc.Row(
                [dbc.Col(viz01_video_player(), width=12)],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(viz02_pipeline_waterfall(), width=6),
                    dbc.Col(viz04_modality_gates(), width=6),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(viz03_frame_gallery(), width=8),
                    dbc.Col(viz05_qa_interface(), width=4),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(viz06_processing_queue(), width=6),
                    dbc.Col(viz07_evaluation_scores(), width=6),
                ],
                className="mb-2",
            ),
        ]
    )
