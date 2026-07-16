"""Live sentiment demo — replaces the old Django+React app with a single file.

Paste comments or point at a YouTube video, and compare predictions from the
thesis models (TF-IDF, LogReg, SVM, ensemble, neuro-fuzzy) side by side. Runs
fully locally against the pinned model artifacts in backend/models/; comment
extraction scrapes YouTube directly (no API key, no cost).

Run from the repo root:
    streamlit run demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine  # noqa: E402
from youtube_fetch import fetch_comments  # noqa: E402

CLASSICAL_MODELS = ["tfidf", "logreg", "svm", "ensemble", "fuzzy_ensemble"]
LABELS = ["Positive", "Neutral", "Negative"]
LABEL_COLORS = {"Positive": "#2e7d32", "Neutral": "#616161", "Negative": "#c62828"}
LABEL_ICONS = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_engine(name: str):
    return get_sentiment_engine(name)


@st.cache_data(show_spinner="Fetching comments from YouTube...", ttl=3600)
def fetch_comments_cached(url: str, max_results: int, sort_by: str) -> list[dict]:
    return fetch_comments(url, max_results=max_results, sort_by=sort_by)


def classify_batch(name: str, texts: list[str]) -> list:
    engine = load_engine(name)
    return [coerce_sentiment_result(r, name) for r in engine.batch_analyze(texts)]


def label_badge(label: str) -> str:
    color = LABEL_COLORS.get(label, "#455a64")
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:10px;font-size:0.85em">{label}</span>'
    )


def show_per_comment(comments: list[str], selected: list[str]) -> None:
    """Detailed view: one section per comment with per-model probabilities."""
    results = {name: classify_batch(name, comments) for name in selected}
    for i, comment in enumerate(comments):
        st.divider()
        st.markdown(f"**Comment:** {comment}")
        cols = st.columns(len(selected))
        for col, name in zip(cols, selected):
            res = results[name][i]
            with col:
                st.markdown(f"**{name}**")
                st.markdown(label_badge(res.label), unsafe_allow_html=True)
                st.caption(f"confidence {res.score:.1%}")
        df = pd.DataFrame({name: results[name][i].probs for name in selected})
        st.bar_chart(df.reindex(LABELS), height=220, stack=False, use_container_width=True)


def show_batch(comments: list[dict], selected: list[str]) -> None:
    """Aggregate view for a fetched video: distribution + per-comment table."""
    texts = [c["text"] for c in comments]
    results = {name: classify_batch(name, texts) for name in selected}

    st.subheader(f"Sentiment of {len(texts)} comments")

    dist = pd.DataFrame(
        {
            name: pd.Series([r.label for r in results[name]]).value_counts()
            for name in selected
        }
    ).reindex(LABELS).fillna(0).astype(int)
    st.bar_chart(dist, height=260, stack=False, use_container_width=True)

    table = pd.DataFrame(
        {
            "comment": [t if len(t) <= 120 else t[:117] + "..." for t in texts],
            "likes": [c["likes"] for c in comments],
            **{
                name: [f"{LABEL_ICONS[r.label]} {r.label}" for r in results[name]]
                for name in selected
            },
        }
    )
    st.dataframe(table, use_container_width=True, height=420)


st.set_page_config(page_title="YouTube Sentiment Demo", page_icon="🎬", layout="wide")
st.title("🎬 YouTube Comment Sentiment — Live Demo")
st.caption(
    "Classical and computational-intelligence models from the thesis, side by "
    "side. Models and pinned artifacts are the exact ones evaluated in the "
    "thesis (see `backend/results/`). Fully local — no API keys."
)

with st.sidebar:
    st.header("Models")
    selected = [m for m in CLASSICAL_MODELS if st.checkbox(m, value=True)]

tab_url, tab_paste = st.tabs(["🔗 YouTube link", "✍️ Paste comments"])

with tab_url:
    url = st.text_input(
        "YouTube video URL or ID:",
        placeholder="https://www.youtube.com/watch?v=jNQXAC9IVRw",
    )
    col1, col2 = st.columns(2)
    with col1:
        max_comments = st.slider("Max comments", 10, 200, 50, step=10)
    with col2:
        sort_by = st.selectbox("Sort by", ["top", "new"])
    if st.button("Fetch & analyze", type="primary", key="fetch"):
        if not selected:
            st.warning("Select at least one model.")
        elif not url.strip():
            st.warning("Enter a YouTube URL or video ID.")
        else:
            try:
                comments = fetch_comments_cached(url.strip(), max_comments, sort_by)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
            else:
                show_batch(comments, selected)

with tab_paste:
    comments_raw = st.text_area(
        "Paste one or more comments (one per line):",
        height=140,
        placeholder="This video is amazing, I loved every second!\nMeh, it was okay I guess.\nWorst tutorial I've ever seen, total waste of time.",
    )
    if st.button("Analyze", type="primary", key="paste"):
        comments = [c.strip() for c in comments_raw.splitlines() if c.strip()]
        if not comments:
            st.warning("Enter at least one comment.")
        elif not selected:
            st.warning("Select at least one model.")
        else:
            show_per_comment(comments, selected)
