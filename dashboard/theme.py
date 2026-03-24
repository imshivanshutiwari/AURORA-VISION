"""AURORA-VISION Vision Ops Center dark teal theme."""

BG_PRIMARY = "#060a0f"
BG_PANEL = "#080d12"
BG_CARD = "#0a1016"
BORDER = "#0f2018"
ACCENT_TEAL = "#1d9e75"
ACCENT_DIM = "#0a3324"
SUCCESS = "#5dcaa5"
WARNING = "#ba7517"
CRITICAL = "#e24b4a"
INFO = "#378add"
TEXT = "#9fe1cb"
TEXT_DIM = "#1a4a38"
FONT = "JetBrains Mono, monospace"

PLOTLY_THEME = {
    "layout": {
        "paper_bgcolor": BG_CARD,
        "plot_bgcolor": BG_PANEL,
        "font": {"color": TEXT, "family": FONT, "size": 11},
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
        "colorway": [ACCENT_TEAL, INFO, WARNING, CRITICAL, SUCCESS, "#a855f7", "#f97316"],
        "xaxis": {
            "gridcolor": BORDER,
            "linecolor": BORDER,
            "tickfont": {"color": TEXT},
            "titlefont": {"color": TEXT},
        },
        "yaxis": {
            "gridcolor": BORDER,
            "linecolor": BORDER,
            "tickfont": {"color": TEXT},
            "titlefont": {"color": TEXT},
        },
        "legend": {
            "bgcolor": BG_CARD,
            "bordercolor": BORDER,
            "font": {"color": TEXT},
        },
    }
}

CARD_STYLE = {
    "backgroundColor": BG_CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "4px",
    "padding": "12px",
    "marginBottom": "12px",
}

HEADER_STYLE = {
    "backgroundColor": BG_PANEL,
    "borderBottom": f"2px solid {ACCENT_TEAL}",
    "padding": "12px 20px",
    "fontFamily": FONT,
    "color": TEXT,
}

STATUS_BAR_STYLE = {
    "backgroundColor": BG_PRIMARY,
    "borderTop": f"1px solid {BORDER}",
    "padding": "4px 20px",
    "fontFamily": FONT,
    "fontSize": "10px",
    "color": ACCENT_DIM,
}
