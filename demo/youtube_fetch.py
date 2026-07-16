"""Fetch YouTube comments without an API key via youtube-comment-downloader.

Two behaviors are kept deliberately:

- ``language='en'`` forces YouTube to render vote counts in English regardless
  of the host locale; without it, counts come back localized ("2.6 लाख") and
  can't be parsed.
- The sort kwarg name varies across library versions (``sort_by`` vs
  ``sort``), so it's resolved by inspecting the real signature.
"""
from __future__ import annotations

import inspect
import re
from typing import List, Optional


def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:youtube\.com\/watch\?v=)([\w-]{11})",
        r"(?:youtu\.be\/)([\w-]{11})",
        r"(?:youtube\.com\/embed\/)([\w-]{11})",
        r"(?:youtube\.com\/shorts\/)([\w-]{11})",
        r"(?:youtube\.com\/v\/)([\w-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.fullmatch(r"[\w-]{11}", url.strip()):
        return url.strip()
    return None


def _parse_likes(value: object) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if not value:
        return 0
    text = str(value).strip().upper().replace(",", "")
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    if text and text[-1] in multipliers:
        try:
            return int(float(text[:-1]) * multipliers[text[-1]])
        except ValueError:
            return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def fetch_comments(video_url: str, max_results: int = 50, sort_by: str = "top") -> List[dict]:
    """Return a list of {text, author, likes} dicts for a video URL or ID."""
    from youtube_comment_downloader import YoutubeCommentDownloader

    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("That doesn't look like a YouTube URL or video ID.")

    downloader = YoutubeCommentDownloader()
    sig_params = inspect.signature(downloader.get_comments_from_url).parameters
    sort_kwarg = "sort_by" if "sort_by" in sig_params else "sort"
    kwargs = {sort_kwarg: 0 if sort_by == "top" else 1}
    if "language" in sig_params:
        kwargs["language"] = "en"

    comments: List[dict] = []
    try:
        for comment in downloader.get_comments_from_url(
            f"https://www.youtube.com/watch?v={video_id}", **kwargs
        ):
            text = (comment.get("text") or "").strip()
            if not text:
                continue
            comments.append(
                {
                    "text": text,
                    "author": comment.get("author", "Unknown"),
                    "likes": _parse_likes(comment.get("votes", 0)),
                }
            )
            if len(comments) >= max_results:
                break
    except Exception as exc:
        raise RuntimeError(
            "The video could not be fetched. It may be private, region-locked, "
            "have comments disabled, or the scraper may have been blocked."
        ) from exc

    if not comments:
        raise RuntimeError(
            "No comments found. The video may have comments disabled or be age-restricted."
        )
    return comments
