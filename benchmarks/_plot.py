"""matplotlib helpers that emit base64 PNG data URIs for inline HTML embedding."""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_to_data_uri(fig, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def new_fig(w: float = 6.0, h: float = 4.0):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax
