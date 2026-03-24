"""
Page 3 — Agent Flow Monitor (VIZ14-VIZ17)
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

NODE_POSITIONS = {
    "ingestor_node": (0, 0),
    "visual_node": (1, 1),
    "audio_node": (1, 0),
    "ocr_node": (1, -1),
    "fusion_node": (2, 0),
    "qa_node": (3, 0),
    "evaluator_node": (4, 0),
}

EDGES = [
    ("ingestor_node", "visual_node"),
    ("ingestor_node", "audio_node"),
    ("ingestor_node", "ocr_node"),
    ("visual_node", "fusion_node"),
    ("audio_node", "fusion_node"),
    ("ocr_node", "fusion_node"),
    ("fusion_node", "qa_node"),
    ("qa_node", "evaluator_node"),
    ("evaluator_node", "qa_node"),
]


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_THEME["layout"])
    return fig


def viz14_langgraph_flow(
    active_nodes: List[str] = None, completed_nodes: List[str] = None
) -> dbc.Card:
    """VIZ14: LangGraph 7-node flow diagram using Cytoscape-like Plotly."""
    active_nodes = active_nodes or []
    completed_nodes = completed_nodes or []

    fig = go.Figure()

    for src, dst in EDGES:
        x0, y0 = NODE_POSITIONS[src]
        x1, y1 = NODE_POSITIONS[dst]
        is_retry = src == "evaluator_node" and dst == "qa_node"
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(
                    color=WARNING if is_retry else BORDER,
                    width=2 if is_retry else 1,
                    dash="dot" if is_retry else "solid",
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for node, (x, y) in NODE_POSITIONS.items():
        if node in active_nodes:
            color = ACCENT_TEAL
            size = 30
        elif node in completed_nodes:
            color = SUCCESS
            size = 24
        else:
            color = INFO
            size = 22

        short = node.replace("_node", "").upper()
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=size, color=color, line=dict(color=BG_PANEL, width=2)),
                text=[short],
                textposition="bottom center",
                textfont=dict(color=TEXT, size=9, family=FONT),
                name=node,
                hovertemplate=f"<b>{node}</b><extra></extra>",
            )
        )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="LangGraph 7-Node Pipeline Flow", font=dict(size=11, color=TEXT)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        showlegend=False,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "⬡ VIZ14 — LANGGRAPH 7-NODE FLOW",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="langgraph-flow", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def viz15_agent_timeline(
    agent_times: Dict[str, Dict] = None, duration: float = 10.0
) -> dbc.Card:
    """VIZ15: Agent execution Gantt chart."""
    if agent_times is None:
        agent_times = {
            "ingestor_node": {"start": 0, "end": 2.5, "tool_calls": [0.5, 1.0, 2.0]},
            "visual_node": {"start": 2.5, "end": 7.0, "tool_calls": [3.0, 5.0]},
            "audio_node": {"start": 2.5, "end": 5.5, "tool_calls": [3.5]},
            "ocr_node": {"start": 2.5, "end": 4.5, "tool_calls": []},
            "fusion_node": {"start": 7.0, "end": 7.5, "tool_calls": []},
            "qa_node": {"start": 7.5, "end": 9.0, "tool_calls": [8.0]},
            "evaluator_node": {"start": 9.0, "end": 10.0, "tool_calls": []},
        }

    parallel_groups = {
        "visual_node": INFO,
        "audio_node": WARNING,
        "ocr_node": SUCCESS,
    }

    fig = go.Figure()
    agent_list = list(agent_times.keys())

    for agent, timing in agent_times.items():
        dur = timing.get("end", 0) - timing.get("start", 0)
        color = parallel_groups.get(agent, ACCENT_TEAL)
        fig.add_trace(
            go.Bar(
                x=[dur],
                y=[agent.replace("_node", "").upper()],
                base=[timing.get("start", 0)],
                orientation="h",
                marker_color=color,
                name=agent,
                showlegend=False,
                hovertemplate=f"<b>{agent}</b><br>Start: {timing.get('start', 0):.2f}s<br>Dur: {dur:.2f}s<extra></extra>",
            )
        )

        for tc in timing.get("tool_calls", []):
            fig.add_trace(
                go.Scatter(
                    x=[tc],
                    y=[agent.replace("_node", "").upper()],
                    mode="markers",
                    marker=dict(symbol="diamond", size=8, color=CRITICAL),
                    showlegend=False,
                    hovertemplate=f"Tool call @ {tc:.2f}s<extra></extra>",
                )
            )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Agent Decision Timeline (◆ = tool call)", font=dict(size=11, color=TEXT)),
        xaxis_title="Time (s)",
        barmode="overlay",
        height=280,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "⏱ VIZ15 — AGENT TIMELINE",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="agent-timeline", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def viz16_modality_contribution(
    query_contributions: List[Dict] = None
) -> dbc.Card:
    """VIZ16: Stacked bar chart of modality contributions per query."""
    if query_contributions is None:
        query_contributions = [
            {"query": "What action?", "visual": 0.65, "audio": 0.20, "text": 0.15},
            {"query": "Who speaks?", "visual": 0.15, "audio": 0.70, "text": 0.15},
            {"query": "What text?", "visual": 0.20, "audio": 0.10, "text": 0.70},
            {"query": "Summarize", "visual": 0.40, "audio": 0.35, "text": 0.25},
            {"query": "Timeline?", "visual": 0.45, "audio": 0.30, "text": 0.25},
        ]

    queries = [d["query"] for d in query_contributions]
    visual = [d.get("visual", 0) for d in query_contributions]
    audio = [d.get("audio", 0) for d in query_contributions]
    text = [d.get("text", 0) for d in query_contributions]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Visual", x=queries, y=visual, marker_color=ACCENT_TEAL)
    )
    fig.add_trace(
        go.Bar(name="Audio", x=queries, y=audio, marker_color=INFO)
    )
    fig.add_trace(
        go.Bar(name="OCR/Text", x=queries, y=text, marker_color=WARNING)
    )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Modality Contribution per Query (%)", font=dict(size=11, color=TEXT)),
        barmode="stack",
        yaxis=dict(tickformat=".0%"),
        height=250,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "◑ VIZ16 — MODALITY CONTRIBUTION",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="modality-contribution", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def viz17_latency_radar(
    latencies: Dict[str, float] = None,
    rolling_avg: Dict[str, float] = None,
) -> dbc.Card:
    """VIZ17: Radar chart of per-agent latency vs rolling average."""
    if latencies is None:
        latencies = {
            "Ingestor": 1200,
            "Visual": 2400,
            "Audio": 1800,
            "OCR": 900,
            "Fusion": 300,
            "QA": 1500,
            "Eval": 200,
        }

    if rolling_avg is None:
        rolling_avg = {k: v * 0.85 for k, v in latencies.items()}

    categories = list(latencies.keys())
    current_vals = list(latencies.values())
    avg_vals = list(rolling_avg.values())

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=current_vals + [current_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=f"rgba(29,158,117,0.2)",
            line=dict(color=ACCENT_TEAL),
            name="Current Video",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=f"rgba(55,138,221,0.15)",
            line=dict(color=INFO, dash="dash"),
            name="Rolling Average",
        )
    )

    bottleneck_idx = current_vals.index(max(current_vals))
    fig.add_trace(
        go.Scatterpolar(
            r=[current_vals[bottleneck_idx]],
            theta=[categories[bottleneck_idx]],
            mode="markers",
            marker=dict(size=12, color=CRITICAL, symbol="star"),
            name=f"Bottleneck: {categories[bottleneck_idx]}",
        )
    )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Agent Latency Radar (ms)", font=dict(size=11, color=TEXT)),
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickfont=dict(color=TEXT, size=7),
                gridcolor=BORDER,
            ),
            angularaxis=dict(tickfont=dict(color=TEXT, size=9)),
            bgcolor=BG_PANEL,
        ),
        height=300,
        showlegend=True,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "◉ VIZ17 — LATENCY RADAR",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="latency-radar", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def build_agent_flow_layout() -> html.Div:
    """Assemble the Agent Flow Monitor page."""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(viz14_langgraph_flow(), width=6),
                    dbc.Col(viz15_agent_timeline(), width=6),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(viz16_modality_contribution(), width=6),
                    dbc.Col(viz17_latency_radar(), width=6),
                ],
                className="mb-2",
            ),
        ]
    )
