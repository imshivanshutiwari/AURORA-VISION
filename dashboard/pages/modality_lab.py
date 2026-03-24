"""
Page 2 — Modality Analysis Lab (VIZ08-VIZ13)
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


def viz08_clip_similarity_heatmap(similarity_matrix: np.ndarray = None, query_labels: List[str] = None, n_frames: int = 32) -> dbc.Card:
    """VIZ08: Frame-level CLIP similarity heatmap (frames × queries)."""
    if similarity_matrix is None:
        similarity_matrix = np.random.uniform(0, 1, (n_frames, max(len(query_labels or []), 5))).astype(float)
        similarity_matrix = (similarity_matrix + similarity_matrix.mean()) / 2

    query_labels = query_labels or [f"Query {i+1}" for i in range(similarity_matrix.shape[1])]

    fig = go.Figure(
        go.Heatmap(
            z=similarity_matrix.T.tolist(),
            x=[f"F{i}" for i in range(similarity_matrix.shape[0])],
            y=query_labels,
            colorscale=[[0, TEXT_DIM], [0.5, INFO], [1, ACCENT_TEAL]],
            colorbar=dict(
                tickfont=dict(color=TEXT, size=8),
                title=dict(text="Cosine Sim", font=dict(color=TEXT, size=9)),
            ),
            zmin=0,
            zmax=1,
            hoverongaps=False,
        )
    )
    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="CLIP Frame-Query Similarity Heatmap", font=dict(size=11, color=TEXT)),
        xaxis_title="Frame Index",
        yaxis_title="Query",
        height=280,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "🌡 VIZ08 — CLIP SIMILARITY HEATMAP",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="clip-heatmap", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def viz09_audio_waveform(
    waveform: np.ndarray = None,
    sample_rate: int = 16000,
    transcription_segments: List[Dict] = None,
) -> dbc.Card:
    """VIZ09: Audio waveform + transcription segments overlay."""
    if waveform is None:
        duration = 30.0
        t = np.linspace(0, duration, int(sample_rate * 0.01))
        waveform = np.sin(2 * np.pi * 2 * t) * np.exp(-0.1 * t) * np.random.normal(1, 0.2, len(t))

    time_axis = np.linspace(0, len(waveform) / sample_rate, len(waveform))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis.tolist(),
            y=waveform.tolist(),
            mode="lines",
            line=dict(color=ACCENT_TEAL, width=0.8),
            name="Waveform",
        )
    )

    seg_colors = [INFO, WARNING, SUCCESS, CRITICAL, "#a855f7"]
    for i, seg in enumerate((transcription_segments or [])[:10]):
        color = seg_colors[i % len(seg_colors)]
        opacity = seg.get("confidence", 0.85)
        fig.add_vrect(
            x0=seg.get("start", 0),
            x1=seg.get("end", 1),
            fillcolor=color,
            opacity=opacity * 0.15,
            layer="below",
            line_width=0,
            annotation_text=f"{seg.get('speaker', 'SPK')}: {seg.get('text', '')[:20]}",
            annotation_position="top left",
            annotation_font=dict(size=8, color=color),
        )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Audio Waveform + Speaker-Tagged Transcription", font=dict(size=11, color=TEXT)),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=250,
        showlegend=False,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "♪ VIZ09 — AUDIO WAVEFORM + TRANSCRIPTION",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="audio-waveform", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def viz10_speaker_diarization(
    speaker_turns: List[Dict] = None, duration: float = 30.0
) -> dbc.Card:
    """VIZ10: Speaker diarization Gantt chart."""
    if speaker_turns is None:
        speaker_turns = [
            {"speaker": "SPEAKER_00", "start": 0, "end": 8, "text": "Hello everyone..."},
            {"speaker": "SPEAKER_01", "start": 8, "end": 18, "text": "Thank you for..."},
            {"speaker": "SPEAKER_00", "start": 18, "end": 25, "text": "As you can see..."},
            {"speaker": "SPEAKER_01", "start": 25, "end": 30, "text": "That concludes..."},
        ]

    speakers = sorted(set(t.get("speaker", "SPK") for t in speaker_turns))
    colors = [ACCENT_TEAL, INFO, WARNING, SUCCESS, CRITICAL, "#a855f7"]

    fig = go.Figure()
    for turn in speaker_turns:
        spk = turn.get("speaker", "SPK")
        spk_idx = speakers.index(spk) if spk in speakers else 0
        color = colors[spk_idx % len(colors)]
        fig.add_trace(
            go.Bar(
                x=[turn.get("end", 0) - turn.get("start", 0)],
                y=[spk],
                base=[turn.get("start", 0)],
                orientation="h",
                marker_color=color,
                marker_line=dict(color=BG_PANEL, width=1),
                text=turn.get("text", "")[:30],
                textposition="inside",
                insidetextanchor="start",
                name=spk,
                showlegend=spk not in [t.name for t in fig.data if hasattr(t, "name")],
                hovertemplate=f"<b>{spk}</b><br>Start: %{{base:.1f}}s<br>Dur: %{{x:.1f}}s<br>{turn.get('text', '')}<extra></extra>",
            )
        )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Speaker Diarization Timeline", font=dict(size=11, color=TEXT)),
        xaxis_title="Time (s)",
        barmode="overlay",
        height=200,
        showlegend=True,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "◎ VIZ10 — SPEAKER DIARIZATION",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="speaker-diarization", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def viz11_ocr_text_map(frame_texts: List[List[str]] = None, n_frames: int = 32) -> dbc.Card:
    """VIZ11: OCR text frequency across frames (persistent vs transient)."""
    if frame_texts is None:
        frame_texts = []
        persistent_texts = ["AURORA-VISION", "Breaking News", "CNN LIVE"]
        for i in range(n_frames):
            ft = []
            for pt in persistent_texts:
                if i % 2 == 0:
                    ft.append(pt)
            if i % 5 == 0:
                ft.append(f"Transient_{i}")
            frame_texts.append(ft)

    all_texts = {}
    for frame_idx, texts in enumerate(frame_texts):
        for t in texts:
            if t not in all_texts:
                all_texts[t] = []
            all_texts[t].append(frame_idx)

    n_total = len(frame_texts)
    threshold = 3

    fig = go.Figure()
    for text, frames_list in sorted(all_texts.items(), key=lambda x: -len(x[1]))[:15]:
        is_persistent = len(frames_list) >= threshold
        color = ACCENT_TEAL if is_persistent else WARNING
        dash_style = "solid" if is_persistent else "dot"
        y_val = [1 if i in frames_list else 0 for i in range(n_total)]
        fig.add_trace(
            go.Scatter(
                x=list(range(n_total)),
                y=[i * 1.2 + 0.1 for i in [list(all_texts.keys()).index(text)] * n_total],
                mode="lines+markers",
                line=dict(color=color, width=2, dash=dash_style),
                marker=dict(color=[color if v else "rgba(0,0,0,0)" for v in y_val], size=6),
                name=text[:20],
                hovertemplate=f"<b>{text[:30]}</b><br>Appears {len(frames_list)} frames<extra></extra>",
            )
        )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="OCR Text Persistence Map (solid=persistent, dotted=transient)", font=dict(size=10, color=TEXT)),
        xaxis_title="Frame Index",
        height=250,
        showlegend=True,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "Ⓣ VIZ11 — OCR TEXT MAP",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="ocr-text-map", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def viz12_swin_attention(
    attention_map: np.ndarray = None, frame_image: Any = None
) -> dbc.Card:
    """VIZ12: Swin Transformer attention map visualization."""
    if attention_map is None:
        n_patches = 196
        attention_map = np.random.dirichlet(np.ones(n_patches), size=1)[0].reshape(14, 14)

    fig = go.Figure(
        go.Heatmap(
            z=attention_map.tolist(),
            colorscale=[[0, BG_PANEL], [0.5, INFO], [1, ACCENT_TEAL]],
            colorbar=dict(
                tickfont=dict(color=TEXT, size=8),
                title=dict(text="Attention", font=dict(color=TEXT, size=9)),
            ),
            hoverongaps=False,
        )
    )
    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Swin Transformer Attention Map (14×14 patches)", font=dict(size=11, color=TEXT)),
        height=280,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "⊞ VIZ12 — SWIN ATTENTION MAP",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="swin-attention", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def viz13_cross_modal_attention(
    attention_matrices: Dict[str, np.ndarray] = None
) -> dbc.Card:
    """VIZ13: Cross-modal attention weight matrices (3×3 grid)."""
    if attention_matrices is None:
        seq_len = 16
        attention_matrices = {
            "visual→audio": np.random.dirichlet(np.ones(seq_len), seq_len),
            "visual→text": np.random.dirichlet(np.ones(seq_len), seq_len),
            "audio→visual": np.random.dirichlet(np.ones(seq_len), seq_len),
        }

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"<span style='color:{TEXT};font-size:9px'>{k}</span>" for k in attention_matrices.keys()],
        horizontal_spacing=0.08,
    )

    colors_list = [
        [[0, BG_PANEL], [1, ACCENT_TEAL]],
        [[0, BG_PANEL], [1, INFO]],
        [[0, BG_PANEL], [1, WARNING]],
    ]

    for i, (name, mat) in enumerate(attention_matrices.items()):
        fig.add_trace(
            go.Heatmap(
                z=mat.tolist(),
                colorscale=colors_list[i],
                showscale=False,
                name=name,
                hoverongaps=False,
            ),
            row=1,
            col=i + 1,
        )

    _apply_theme(fig)
    fig.update_layout(
        title=dict(text="Cross-Modal Attention Weights", font=dict(size=11, color=TEXT)),
        height=280,
    )

    return dbc.Card(
        [
            dbc.CardHeader(
                "⊗ VIZ13 — CROSS-MODAL ATTENTION",
                style={"color": ACCENT_TEAL, "backgroundColor": BG_PANEL, "borderBottom": f"1px solid {BORDER}"},
            ),
            dbc.CardBody(dcc.Graph(id="cross-modal-attn", figure=fig, config={"displayModeBar": False})),
        ],
        style=CARD_STYLE,
    )


def build_modality_lab_layout() -> html.Div:
    """Assemble the Modality Analysis Lab page."""
    return html.Div(
        [
            dbc.Row([dbc.Col(viz08_clip_similarity_heatmap(), width=12)], className="mb-2"),
            dbc.Row(
                [
                    dbc.Col(viz09_audio_waveform(), width=6),
                    dbc.Col(viz10_speaker_diarization(), width=6),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(viz11_ocr_text_map(), width=6),
                    dbc.Col(viz12_swin_attention(), width=3),
                    dbc.Col(viz13_cross_modal_attention(), width=3),
                ],
                className="mb-2",
            ),
        ]
    )
