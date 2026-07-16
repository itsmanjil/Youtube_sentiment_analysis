"""Shared helpers for the research notebooks.

Every notebook loads already-generated artifacts from backend/results and
backend/figures rather than recomputing them, so notebooks open fast and
don't require torch/transformers to be installed.
"""
import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, SVG, Markdown, display

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
RESULTS = BACKEND / "results"
FIGURES = BACKEND / "figures"
DATA = BACKEND / "data"
RESEARCH = BACKEND / "research"


def load_json(relpath, root=RESULTS):
    path = root / relpath
    return json.loads(path.read_text(encoding="utf-8"))


def show_md(relpath, root=RESULTS):
    path = root / relpath
    if not path.exists():
        display(Markdown(f"*(missing: `{path.relative_to(ROOT)}`)*"))
        return
    display(Markdown(path.read_text(encoding="utf-8")))


def show_fig(name, root=FIGURES, width=650):
    path = root / name
    if not path.exists():
        display(Markdown(f"*(missing figure: `{path.relative_to(ROOT)}`)*"))
        return
    if path.suffix == ".svg":
        display(SVG(filename=str(path)))
    else:
        display(Image(filename=str(path), width=width))


def show_thesis_fig(name, width=650):
    show_fig(name, root=FIGURES / "thesis", width=width)


def metrics_table(results_dict, metric_keys=None):
    """results_dict: {model_name: {metric: value, ...}}."""
    df = pd.DataFrame(results_dict).T
    if metric_keys:
        df = df[metric_keys]
    return df
