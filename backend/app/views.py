
import json
import math
import re
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

from googleapiclient.errors import HttpError

from django.conf import settings
from django.db import connection
from django.db.utils import Error as DjangoDBError
from django.http import JsonResponse
from django.utils import timezone
from django_q.tasks import async_task
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle

from .models import AnalysisJob, YouTubeVideo, YouTubeComment, YouTubeAnalysis
from .youtube_fetcher import YouTubeFetcher
from .youtube_scraper import YouTubeScraper
from .youtube_preprocessor import YouTubePreprocessor
from .aspect_mining import extract_aspect_sentiment
from src.utils import (
    aggregate_confidence_stats,
    bootstrap_confidence_intervals,
    build_hourly_sentiment,
    confidence_from_probs,
    entropy_from_probs,
    get_runtime_artifact_metadata,
)
from src.utils.config import Config
from src.sentiment import (
    coerce_sentiment_result,
    get_sentiment_engine,
)

# Use absolute paths from Django project root
BASE_DIR = Path(__file__).resolve().parent.parent

_MODEL_ALIASES = {
    "ci_ensemble": "ensemble",
    "meta": "meta_learner",
    "meta-learner": "meta_learner",
    "stacking": "meta_learner",
    "fuzzy": "fuzzy_ensemble",
    "fuzzy-ensemble": "fuzzy_ensemble",
    "deberta-v3": "deberta_v3",
}

# Only presets with a fine-tuned checkpoint under backend/models/transformers/
# are exposed here — bert/roberta/modernbert/xlm_v/mdeberta_v3 have no shipped
# artifact and would only ever fail with "no fine-tuned checkpoint found"
# (see TransformerSentimentEngine.__init__). Training a new preset is still
# possible via research/transformers/train_encoder.py and its own registry
# (research/transformers/model_registry.py); once a checkpoint exists under
# backend/models/transformers/<preset>, add it back here.
_TRANSFORMER_MODELS = {
    "deberta_v3",
}

# Canonical sentiment_model names the factory (src/sentiment/factory.py) can
# construct, after _normalize_sentiment_model resolves view-layer aliases
# (meta/stacking/fuzzy/deberta-v3/...) to these. Static rather than probed via
# list_available_engines(): that function additionally gates transformer/
# hybrid_dl names on torch/transformers actually being importable, which is
# the right check at engine-construction time but would make this synchronous
# pre-check reject a canonical, mockable model name in an environment that
# simply doesn't have those optional deps installed (e.g. the test suite,
# which mocks get_sentiment_engine directly for transformer-preset tests).
# An unavailable-dependency request still fails cleanly via the ImportError
# handling in _execute_analysis_job — this check only catches typos/garbage.
_KNOWN_SENTIMENT_MODELS = {
    "tfidf", "logreg", "svm",
    "ensemble", "meta_learner", "fuzzy_ensemble",
    "hybrid_dl",
} | _TRANSFORMER_MODELS


def _coerce_model_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [
            model.strip().lower()
            for model in value.split(",")
            if model.strip()
        ]
    if isinstance(value, list):
        return [
            str(model).strip().lower()
            for model in value
            if str(model).strip()
        ]
    return None


def _coerce_ensemble_weights(value):
    if value is None:
        return None, None

    def unwrap_weights(payload):
        if isinstance(payload, dict):
            nested = payload.get("weights")
            if isinstance(nested, (dict, list, tuple)):
                return nested, "weights"
        return payload, None

    def coerce_structured_weights(payload, source):
        weights, suffix = unwrap_weights(payload)
        if not isinstance(weights, (dict, list, tuple)):
            return None, None
        # Reject non-numeric, non-finite (NaN/inf), negative, or all-zero
        # weights here so the caller returns a clean 400. Otherwise a NaN
        # weight sails past the ensemble's `total <= 0` guard (NaN compares
        # False to everything) and crashes the analysis job with an opaque 500.
        raw_values = weights.values() if isinstance(weights, dict) else weights
        numeric_values = []
        for candidate in raw_values:
            try:
                numeric = float(candidate)
            except (TypeError, ValueError):
                return None, None
            if not math.isfinite(numeric) or numeric < 0:
                return None, None
            numeric_values.append(numeric)
        if not numeric_values or not any(value > 0 for value in numeric_values):
            return None, None
        if suffix:
            source = f"{source}:{suffix}"
        return weights, source

    if isinstance(value, (dict, list, tuple)):
        return coerce_structured_weights(value, "request")
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None, None
        try:
            payload = json.loads(raw)
            return coerce_structured_weights(payload, "json")
        except json.JSONDecodeError:
            return None, None
    return None, None


def _has_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


class _InvalidParam(ValueError):
    """Raised for a malformed/out-of-range request parameter; caught and
    turned into a 400 response rather than propagating as an unhandled 500."""


def _parse_bounded_int(value, default, *, minimum, maximum, name):
    if value is None:
        value = default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise _InvalidParam(f"{name} must be an integer.")
    if parsed < minimum or parsed > maximum:
        raise _InvalidParam(
            f"{name} must be between {minimum} and {maximum} (got {parsed})."
        )
    return parsed


def _parse_bounded_float(value, default, *, minimum, maximum, name):
    if value is None:
        value = default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise _InvalidParam(f"{name} must be a number.")
    if parsed < minimum or parsed > maximum:
        raise _InvalidParam(
            f"{name} must be between {minimum} and {maximum} (got {parsed})."
        )
    return parsed


def _parse_choice(value, default, *, choices, name):
    # Mirrors _parse_bounded_int/_parse_bounded_float above but for the
    # free-string params (emoji_mode, the fuzzy engine's mf_type/
    # defuzz_method/t_norm/t_conorm, ensemble_weights_optimization) that
    # used to be accepted as-is with no validation. A typo (e.g.
    # emoji_mode="covnert") wasn't rejected -- it silently fell through to
    # whatever "else" branch the downstream code happened to have, picking a
    # real-but-unintended value (convert_emojis's else-branch is "keep";
    # ensemble_engine's weights_optimization check silently means "pso") with
    # no indication to the caller that their input was ignored.
    if value is None:
        value = default
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise _InvalidParam(
            f"{name} must be one of {sorted(choices)} (got {value!r})."
        )
    return normalized


def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _normalize_sentiment_model(value):
    normalized = str(value or "logreg").lower().strip()
    return _MODEL_ALIASES.get(normalized, normalized)


def _is_transformer_model(model_name):
    return model_name in _TRANSFORMER_MODELS


def _get_processing_profile(model_name):
    return "transformer" if _is_transformer_model(model_name) else "classical"


def _get_model_family(model_name):
    if _is_transformer_model(model_name):
        return "transformer"
    if model_name in ("ensemble", "meta_learner", "fuzzy_ensemble"):
        return "ensemble"
    if model_name == "hybrid_dl":
        return "deep_learning"
    return "classical"


# Same patterns as YouTubeFetcher.extract_video_id / YouTubeScraper.extract_video_id
# (app/youtube_fetcher.py, app/youtube_scraper.py) — duplicated here so the view can
# reject a malformed video_url synchronously, before a job is created, without
# instantiating either fetcher (which requires a configured YOUTUBE_API_KEY) or the
# scraper. The fetch step still calls the authoritative extract_video_id itself.
_YOUTUBE_URL_PATTERNS = [
    re.compile(r'(?:youtube\.com\/watch\?v=)([\w-]{11})'),
    re.compile(r'(?:youtu\.be\/)([\w-]{11})'),
    re.compile(r'(?:youtube\.com\/embed\/)([\w-]{11})'),
    re.compile(r'(?:youtube\.com\/v\/)([\w-]{11})'),
]
_BARE_VIDEO_ID = re.compile(r'^[\w-]{11}$')


def _looks_like_youtube_url(url):
    if not isinstance(url, str):
        return False
    if any(pattern.search(url) for pattern in _YOUTUBE_URL_PATTERNS):
        return True
    return bool(_BARE_VIDEO_ID.match(url))



@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([ScopedRateThrottle])
def analyze_youtube_video(request):
    user = request.user

    # Extract parameters
    video_url = request.data.get("video_url")
    try:
        max_comments = _parse_bounded_int(
            request.data.get("max_comments"), 200, minimum=1, maximum=2000, name="max_comments"
        )
        use_api = _coerce_bool(request.data.get("use_api"), default=True)
        filter_spam = _coerce_bool(request.data.get("filter_spam"), default=True)
        filter_language = _coerce_bool(request.data.get("filter_language"), default=True)
        bootstrap_samples = _parse_bounded_int(
            request.data.get("bootstrap_samples"),
            500,
            minimum=1,
            maximum=2000,
            name="bootstrap_samples",
        )
        random_seed = _parse_bounded_int(
            request.data.get("random_seed"), 42, minimum=0, maximum=2**31 - 1, name="random_seed"
        )
        aspect_top_n = _parse_bounded_int(
            request.data.get("aspect_top_n"), 12, minimum=1, maximum=200, name="aspect_top_n"
        )
        aspect_min_freq = _parse_bounded_int(
            request.data.get("aspect_min_freq"), 3, minimum=1, maximum=1000, name="aspect_min_freq"
        )
        confidence_threshold = _parse_bounded_float(
            request.data.get("confidence_threshold"),
            0.6,
            minimum=0.0,
            maximum=1.0,
            name="confidence_threshold",
        )
        fuzzy_alpha_cut = _parse_bounded_float(
            request.data.get("fuzzy_alpha_cut"),
            0.0,
            minimum=0.0,
            maximum=1.0,
            name="fuzzy_alpha_cut",
        )
        fuzzy_resolution = _parse_bounded_int(
            request.data.get("fuzzy_resolution"),
            100,
            minimum=10,
            maximum=1000,
            name="fuzzy_resolution",
        )
        # These mirror the <select> options Search.jsx actually offers (the
        # canonical valid-value list); an unrecognized value now gets a
        # clear 400 instead of being silently reinterpreted downstream.
        emoji_mode = _parse_choice(
            request.data.get("emoji_mode"),
            "convert",
            choices={"remove", "convert", "keep"},
            name="emoji_mode",
        )
        ensemble_weights_optimization = _parse_choice(
            request.data.get("ensemble_weights_optimization"),
            "pso",
            choices={"pso", "nsga2"},
            name="ensemble_weights_optimization",
        )
        fuzzy_mf_type = _parse_choice(
            request.data.get("fuzzy_mf_type"),
            "gaussian",
            choices={"triangular", "trapezoidal", "gaussian"},
            name="fuzzy_mf_type",
        )
        fuzzy_defuzz_method = _parse_choice(
            request.data.get("fuzzy_defuzz_method"),
            "centroid",
            choices={"centroid", "bisector", "mom", "som", "lom", "weighted_average"},
            name="fuzzy_defuzz_method",
        )
        fuzzy_t_norm = _parse_choice(
            request.data.get("fuzzy_t_norm"),
            "min",
            choices={"min", "product", "lukasiewicz"},
            name="fuzzy_t_norm",
        )
        fuzzy_t_conorm = _parse_choice(
            request.data.get("fuzzy_t_conorm"),
            "max",
            choices={"max", "prob_sum", "bounded_sum"},
            name="fuzzy_t_conorm",
        )
    except _InvalidParam as exc:
        return Response({"msg": str(exc)}, status=400)

    sentiment_model = _normalize_sentiment_model(request.data.get("sentiment_model", "logreg"))
    # calibration_profile isn't a closed enum like the fields above -- it's
    # effectively a disable-flag with several accepted spellings ("off",
    # "none", "disabled", "raw"; see transformer_engine.py); anything else,
    # typo or not, means "enabled/auto", which is already the sensible
    # default, so a strict choice list isn't needed here.
    calibration_profile = str(request.data.get("calibration_profile", "auto") or "auto")
    ensemble_models = _coerce_model_list(request.data.get("ensemble_models"))
    ensemble_weights_input = request.data.get("ensemble_weights")
    meta_learner_path = request.data.get("meta_learner_path")
    meta_learner_models = _coerce_model_list(request.data.get("meta_learner_models"))
    fuzzy_models = _coerce_model_list(request.data.get("fuzzy_models"))
    model_comparison = request.data.get("model_comparison")

    if ensemble_models is None:
        ensemble_models = ["logreg", "svm", "tfidf"]
    ensemble_weights, ensemble_weights_source = _coerce_ensemble_weights(
        ensemble_weights_input
    )
    if _has_value(ensemble_weights_input) and ensemble_weights is None:
        return Response(
            {"msg": "Invalid ensemble_weights. Provide inline JSON weights."},
            status=400,
        )
    if _has_value(meta_learner_path):
        return Response(
            {
                "msg": (
                    "meta_learner_path overrides are not supported from API requests. "
                    "Use the server-configured meta-learner artifact."
                )
            },
            status=400,
        )
    meta_learner_path = None

    if sentiment_model == "fuzzy_ensemble" and not fuzzy_models:
        fuzzy_models = ["logreg", "svm", "tfidf"]

    if not video_url:
        return Response({"msg": "video_url is required"}, status=400)
    if not _looks_like_youtube_url(video_url):
        return Response({"msg": f"Invalid YouTube URL: {video_url}"}, status=400)

    if sentiment_model not in _KNOWN_SENTIMENT_MODELS:
        return Response(
            {
                "msg": (
                    f"Invalid sentiment_model: '{sentiment_model}'. "
                    f"Available models: {sorted(_KNOWN_SENTIMENT_MODELS)}"
                )
            },
            status=400,
        )

    # Every validated input the background job needs, since it runs outside
    # this request/response cycle (a plain function argument list would work
    # for the synchronous/test path, but the async path needs this captured
    # as data anyway to hand to the Django-Q2 worker).
    params = {
        "video_url": video_url,
        "max_comments": max_comments,
        "use_api": use_api,
        "filter_spam": filter_spam,
        "filter_language": filter_language,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "aspect_top_n": aspect_top_n,
        "aspect_min_freq": aspect_min_freq,
        "confidence_threshold": confidence_threshold,
        "fuzzy_alpha_cut": fuzzy_alpha_cut,
        "fuzzy_resolution": fuzzy_resolution,
        "emoji_mode": emoji_mode,
        "sentiment_model": sentiment_model,
        "calibration_profile": calibration_profile,
        "ensemble_models": ensemble_models,
        "ensemble_weights_optimization": ensemble_weights_optimization,
        "ensemble_weights": ensemble_weights,
        "ensemble_weights_source": ensemble_weights_source,
        "meta_learner_models": meta_learner_models,
        "fuzzy_models": fuzzy_models,
        "fuzzy_mf_type": fuzzy_mf_type,
        "fuzzy_defuzz_method": fuzzy_defuzz_method,
        "fuzzy_t_norm": fuzzy_t_norm,
        "fuzzy_t_conorm": fuzzy_t_conorm,
        "model_comparison": model_comparison,
    }

    job = AnalysisJob.objects.create(user=user, request_params=params)

    if settings.ANALYSIS_RUN_SYNC:
        _execute_analysis_job(job.id)
        job.refresh_from_db()
        if job.status == AnalysisJob.STATUS_DONE:
            return Response(job.result)
        return Response(
            {"msg": job.error_message or "Analysis failed due to an internal error."},
            status=job.error_status or 500,
        )

    # Dispatched to a Django-Q2 worker process (see Q_CLUSTER in
    # core/settings.py) rather than run in this request/response cycle —
    # requires `python manage.py qcluster` running; see README.md. The
    # worker manages its own DB connections per task, unlike the old
    # daemon-thread approach this replaced.
    async_task(_execute_analysis_job, job.id)
    return Response(
        {
            "msg": "Analysis started",
            "job_id": job.id,
            "status": job.status,
        },
        status=202,
    )


def _fail_job(job, message, http_status):
    job.status = AnalysisJob.STATUS_FAILED
    job.error_message = message
    job.error_status = http_status
    job.save(update_fields=["status", "error_message", "error_status", "updated_at"])


def _fetch_comments(job, p, user):
    """Step 1: Fetch video metadata + raw comments via the YouTube API or
    scraper, validating the result. Returns (video_id, video_metadata,
    comments_raw) on success, or None if the job was already failed — the
    caller must return immediately in that case."""
    video_url = p["video_url"]
    max_comments = p["max_comments"]
    use_api = p["use_api"]

    logger.debug("Fetching comments from %s", video_url)
    if use_api:
        try:
            fetcher = YouTubeFetcher()
            video_id = fetcher.extract_video_id(video_url)
            if video_id is None:
                _fail_job(job, f"Invalid YouTube URL: {video_url}", 400)
                return None
            video_metadata = fetcher.fetch_video_metadata(video_id)
            comments_raw = fetcher.fetch_comments(video_url, max_results=max_comments)
        except HttpError as e:
            # Never surface str(e)/repr(e) here: HttpError.__str__ always
            # embeds the full request URL (self.uri), which for this
            # client includes `key=<YOUTUBE_API_KEY>` as a query
            # parameter — that would leak the shared API key to whichever
            # user's request happened to fail.
            status_code = getattr(e.resp, "status", None)
            try:
                error_details = json.loads(e.content).get('error', {})
                reason = error_details.get('errors', [{}])[0].get('reason')
                message = error_details.get('message') or (
                    f"YouTube API request failed with status {status_code}."
                )

                if reason == 'quotaExceeded':
                    _fail_job(job, "YouTube API daily quota exceeded. Please try again tomorrow or use scraper mode (use_api: false).", 429); return None
                elif reason == 'developerKeyInvalid':
                    _fail_job(job, "The provided YOUTUBE_API_KEY is invalid. Please check your .env file and ensure it is correct.", 401); return None
                elif reason == 'commentsDisabled':
                    _fail_job(job, "Comments are disabled for this video.", 403); return None
                elif status_code == 404:
                    _fail_job(job, "Video not found. Please check the URL.", 404); return None
                else:
                    _fail_job(job, f"A YouTube API error occurred: {message}", 502); return None
            except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
                logger.warning(
                    "Unparseable YouTube API error response (status=%s)", status_code
                )
                _fail_job(
                    job,
                    f"A YouTube API error occurred (status {status_code}).",
                    502,
                )
                return None
        except ValueError as e:
            # Local configuration error raised by YouTubeFetcher.__init__
            # (e.g. missing YOUTUBE_API_KEY) — safe to surface verbatim,
            # it never touches the network or embeds request details.
            _fail_job(job, str(e), 502)
            return None
        except Exception:
            logger.exception(
                "Unexpected error using the YouTube API client for user=%s video_url=%s",
                getattr(user, "id", None),
                video_url,
            )
            _fail_job(
                job,
                "An unexpected error occurred with the YouTube API client. Please ensure your YOUTUBE_API_KEY is correctly set in the .env file.",
                502,
            )
            return None
    else:
        try:
            scraper = YouTubeScraper()
            video_id = scraper.extract_video_id(video_url)
            if video_id is None:
                _fail_job(job, f"Invalid YouTube URL: {video_url}", 400)
                return None
            video_metadata = scraper.fetch_video_metadata(video_id)
            comments_raw = scraper.fetch_comments(video_url, max_results=max_comments)
        except (RuntimeError, ImportError) as e:
            # YouTubeScraper deliberately raises only these two exception
            # types, both with pre-sanitized, human-authored messages
            # (missing youtube-comment-downloader install; comments
            # disabled/private/region-locked/blocked) — see
            # youtube_scraper.py, which itself takes care never to let
            # raw yt-dlp/downloader exception text (which can embed local
            # paths or other internals) propagate this far. Safe to
            # surface verbatim.
            _fail_job(job, f"Scraper error: {str(e)}", 502)
            return None
        except Exception:
            # Anything else is an exception type the scraper module
            # never intentionally raises (see above) — treat it the same
            # as the API-mode catch-all a few lines up: log the real
            # detail server-side, return a generic client-safe message.
            logger.exception(
                "Unexpected scraper error for user=%s video_url=%s",
                getattr(user, "id", None),
                video_url,
            )
            _fail_job(
                job,
                "An unexpected error occurred while scraping YouTube. Please try again later.",
                502,
            )
            return None

    if not video_metadata:
        _fail_job(job, "Video not found. It may be private, deleted, or the URL is incorrect.", 404)
        return None

    if not comments_raw:
        _fail_job(job, "No comments found for this video", 404)
        return None

    logger.debug("Fetched %s raw comments", len(comments_raw))
    return video_id, video_metadata, comments_raw


def _get_or_update_video(video_id, video_metadata):
    """Step 2: Save or update video metadata. Returns the YouTubeVideo."""
    video, created = YouTubeVideo.objects.get_or_create(
        video_id=video_id,
        defaults={
            'title': video_metadata['title'],
            'description': video_metadata.get('description', ''),
            'channel_name': video_metadata['channel'],
            'channel_id': video_metadata.get('channel_id', ''),
            'published_at': video_metadata['published_at'],
            'view_count': video_metadata.get('view_count', 0),
            'like_count': video_metadata.get('like_count', 0),
            'comment_count': video_metadata.get('comment_count', 0),
            'thumbnail_url': video_metadata.get('thumbnail_url', '')
        }
    )

    if not created:
        # Update metadata if video already exists
        video.view_count = video_metadata.get('view_count', video.view_count)
        video.like_count = video_metadata.get('like_count', video.like_count)
        video.comment_count = video_metadata.get('comment_count', video.comment_count)
        video.save()

    return video


def _preprocess_comments(job, p, sentiment_model, comments_raw):
    """Step 3: Preprocess + filter raw comments. Returns (processed_comments,
    filter_stats, processing_profile) on success, or None if the job was
    already failed — the caller must return immediately in that case."""
    logger.debug("Preprocessing comments")
    preprocessor = YouTubePreprocessor()
    processing_profile = _get_processing_profile(sentiment_model)
    processed_comments, filter_stats = preprocessor.batch_preprocess(
        comments_raw,
        profile=processing_profile,
        emoji_mode=p["emoji_mode"],
        check_spam=p["filter_spam"],
        check_lang=p["filter_language"]
    )

    if not processed_comments:
        _fail_job(job, "All comments were filtered out. Try different filter settings.", 400)
        return None

    logger.debug("Processed %s comments after filtering", len(processed_comments))
    return processed_comments, filter_stats, processing_profile


def _build_engine_kwargs(sentiment_model, p):
    """Step 4a: Build the get_sentiment_engine() kwargs for the requested
    model family."""
    engine_kwargs = {}
    if sentiment_model == "ensemble":
        engine_kwargs = {
            "base_models": p["ensemble_models"],
            "weights": p["ensemble_weights"],
            "weights_optimization": p["ensemble_weights_optimization"],
        }
    elif sentiment_model == "meta_learner":
        # meta_learner_path is always None here: enforced at the view layer
        # (meta_learner_path overrides are rejected there with a 400 before
        # a job is ever created).
        if p["meta_learner_models"]:
            engine_kwargs["base_models"] = p["meta_learner_models"]
    elif sentiment_model == "fuzzy_ensemble":
        engine_kwargs = {
            "base_models": p["fuzzy_models"],
            "mf_type": p["fuzzy_mf_type"] or "gaussian",
            "defuzz_method": p["fuzzy_defuzz_method"] or "centroid",
            "t_norm": p["fuzzy_t_norm"] or "min",
            "t_conorm": p["fuzzy_t_conorm"] or "max",
            "alpha_cut": p["fuzzy_alpha_cut"],
            "resolution": p["fuzzy_resolution"],
            "confidence_threshold": p["confidence_threshold"],
        }
    elif _is_transformer_model(sentiment_model):
        engine_kwargs = {
            "calibration_profile": p["calibration_profile"],
        }
    return engine_kwargs


def _run_sentiment_analysis(job, sentiment_model, engine_kwargs, processed_comments, user):
    """Step 4b: Construct the sentiment engine and run batch_analyze,
    mutating processed_comments in place with sentiment/score/probs/
    confidence. Returns the engine on success, or None if the job was
    already failed — the caller must return immediately in that case."""
    logger.debug("Running sentiment analysis using %s", sentiment_model)
    try:
        engine = get_sentiment_engine(sentiment_model, **engine_kwargs)
    except ValueError as exc:
        # "Invalid engine type: ..." — safe, no server-side paths.
        _fail_job(job, str(exc), 400)
        return None
    except (RuntimeError, ImportError, FileNotFoundError):
        # These can embed absolute server-side model/vectorizer paths
        # (see FileNotFoundError raised by e.g. LogRegSentimentEngine.__init__,
        # and src.sentiment.engines.artifact_utils.format_model_load_error) —
        # log the real detail server-side and return a generic message.
        logger.exception(
            "Failed to construct sentiment engine %s for user=%s",
            sentiment_model,
            getattr(user, "id", None),
        )
        _fail_job(
            job,
            f"The '{sentiment_model}' model is temporarily unavailable. Please try a different model or try again later.",
            400,
        )
        return None

    batch_results = engine.batch_analyze(
        [item['processed_text'] for item in processed_comments]
    )
    for item, raw_result in zip(processed_comments, batch_results):
        result = coerce_sentiment_result(raw_result, sentiment_model)
        item['sentiment'] = result.label
        item['sentiment_score'] = result.score
        item['sentiment_probs'] = result.probs
        item['confidence'] = confidence_from_probs(result.probs)

    return engine


def _save_comments(video, processed_comments):
    """Step 5: Save processed comments to the database.

    Batched via bulk_create instead of one update_or_create()/create() per
    comment (up to `max_comments`, i.e. up to 2000 individual queries):
    id-bearing comments are upserted in a single INSERT ... ON CONFLICT DO
    UPDATE, id-less ones in a single plain INSERT. This trades the old
    per-comment error isolation (one malformed comment used to fail only
    itself) for one failure covering the whole batch — acceptable here
    since the analytics step is computed from `processed_comments` in
    memory, not from these rows, so a save failure is already non-fatal to
    the analysis result either way.
    """
    logger.debug("Saving processed comments to database")
    without_id_comments = []
    # Keyed by comment_id so a duplicate id within the same batch keeps
    # only the last occurrence — bulk_create(update_conflicts=True)
    # cannot upsert the same conflict key twice within one statement
    # (unlike the old sequential update_or_create() loop, which just
    # updated the same row twice).
    with_id_comments = {}
    for item in processed_comments:
        # `published_at` is a non-nullable DateTimeField, but scraper mode
        # can yield None for unparseable relative timestamps ("2 days
        # ago"-style strings that don't match any known pattern) — fall
        # back to now() rather than raising IntegrityError on save.
        published_at = item['published_at'] or timezone.now()
        fields = dict(
            video=video,
            text=item['text'],
            author=item['author'],
            likes=item['likes'],
            published_at=published_at,
            is_reply=item['is_reply'],
            sentiment=item['sentiment'],
            sentiment_score=item['sentiment_score'],
            is_spam=item.get('metadata', {}).get('is_spam', False),
            language=item['metadata']['language'],
        )
        # `comment_id` is globally unique in the DB. Coerce a missing/blank
        # id to None (NULL) rather than '' — SQL treats multiple NULLs as
        # distinct for uniqueness, whereas multiple '' comments would
        # collide into a single row and silently overwrite each other.
        comment_id = item.get('comment_id') or None
        if comment_id is None:
            without_id_comments.append(YouTubeComment(comment_id=None, **fields))
        else:
            with_id_comments[comment_id] = YouTubeComment(comment_id=comment_id, **fields)

    try:
        if without_id_comments:
            YouTubeComment.objects.bulk_create(without_id_comments)
        if with_id_comments:
            YouTubeComment.objects.bulk_create(
                list(with_id_comments.values()),
                update_conflicts=True,
                unique_fields=["comment_id"],
                update_fields=[
                    "video", "text", "author", "likes", "published_at",
                    "is_reply", "sentiment", "sentiment_score", "is_spam",
                    "language",
                ],
            )
    except Exception:
        logger.exception(
            "Failed to bulk-save comments for video %s", video.video_id
        )


def _build_analytics(p, processed_comments):
    """Step 6: Compute aggregate analytics from processed_comments."""
    logger.debug("Generating analysis aggregates")
    sentiments = [item['sentiment'] for item in processed_comments]
    sentiment_counts = Counter(sentiments)
    confidences = [item.get('confidence', 0.0) for item in processed_comments]
    confidence_stats = aggregate_confidence_stats(
        confidences,
        threshold=p["confidence_threshold"],
    )
    entropies = [
        entropy_from_probs(item.get('sentiment_probs', {}))
        for item in processed_comments
    ]
    uncertainty_stats = {
        "mean_entropy": round(sum(entropies) / len(entropies), 4) if entropies else 0.0,
        "max_entropy": round(max(entropies), 4) if entropies else 0.0,
        "min_entropy": round(min(entropies), 4) if entropies else 0.0,
        "high_uncertainty_ratio": round(
            sum(1 for e in entropies if e > 0.5) / len(entropies), 4
        ) if entropies else 0.0,
    }
    sentiment_cis = bootstrap_confidence_intervals(
        sentiments,
        n_boot=p["bootstrap_samples"],
        alpha=0.05,
        seed=p["random_seed"],
    )
    aspect_sentiment = extract_aspect_sentiment(
        processed_comments,
        top_n=p["aspect_top_n"],
        min_freq=p["aspect_min_freq"],
    )
    sentiment_timeline = build_hourly_sentiment(processed_comments)

    sentiment_data = {
        'Positive': sentiment_counts.get('Positive', 0),
        'Negative': sentiment_counts.get('Negative', 0),
        'Neutral': sentiment_counts.get('Neutral', 0)
    }

    # Like-weighted sentiment
    like_weighted = []
    for item in processed_comments:
        likes = item['likes']
        if likes > 0:
            like_weighted.append({
                'likes': likes,
                'sentiment': item['sentiment'],
                'text': item['text'][:100],
                'author': item['author']
            })

    like_weighted.sort(key=lambda x: x['likes'], reverse=True)

    # Top words for word clouds
    positive_words = []
    negative_words = []

    for item in processed_comments:
        words = (
            item.get('processed_text_classical')
            or item.get('processed_text')
            or ''
        ).split()
        if item['sentiment'] == 'Positive':
            positive_words.extend(words)
        elif item['sentiment'] == 'Negative':
            negative_words.extend(words)

    top_positive = Counter(positive_words).most_common(50)
    top_negative = Counter(negative_words).most_common(50)

    return {
        "confidence_stats": confidence_stats,
        "uncertainty_stats": uncertainty_stats,
        "sentiment_cis": sentiment_cis,
        "aspect_sentiment": aspect_sentiment,
        "sentiment_timeline": sentiment_timeline,
        "sentiment_data": sentiment_data,
        "like_weighted": like_weighted,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


def _build_analysis_meta(sentiment_model, engine, processing_profile, p, analytics):
    """Step 6b: Build the analysis_meta dict (model-family-specific
    diagnostic metadata)."""
    analysis_meta = {
        "model_family": _get_model_family(sentiment_model),
        "model_artifact": getattr(engine, "model_artifact", None),
        "preprocessing_profile": processing_profile,
        "runtime_artifacts": get_runtime_artifact_metadata(),
        # Per-file sha256 verification against the pinned manifest (see
        # src/utils/runtime_artifacts.py::verify_model_artifact_hash).
        # None = no pinned hash to verify against; True/False = verified
        # match/mismatch for the actual model file this engine loaded.
        "artifact_verified": getattr(engine, "artifact_verified", None),
        "confidence_stats": analytics["confidence_stats"],
        "uncertainty_stats": analytics["uncertainty_stats"],
        "sentiment_confidence_intervals": analytics["sentiment_cis"],
        "aspect_sentiment": analytics["aspect_sentiment"],
        "bootstrap_samples": p["bootstrap_samples"],
        "random_seed": p["random_seed"],
    }
    if _is_transformer_model(sentiment_model):
        analysis_meta["transformer"] = {
            "preset": getattr(engine, "model_preset", None),
            "source": getattr(engine, "model_source", None),
            "artifact": getattr(engine, "model_artifact", None),
            "is_fine_tuned": getattr(engine, "is_fine_tuned", None),
            "max_length": getattr(engine, "max_length", None),
            "device": str(getattr(engine, "device", "")),
            "calibration_profile": getattr(engine, "calibration_profile", p["calibration_profile"]),
            "calibration_applied": getattr(engine, "calibration_applied", False),
            "temperature": getattr(engine, "temperature", None),
            "temperature_artifact_path": getattr(engine, "temperature_artifact_path", None),
        }
    if sentiment_model == "ensemble":
        analysis_meta["ensemble"] = {
            "models": getattr(engine, "requested_models", p["ensemble_models"]),
            "models_used": getattr(engine, "base_models", p["ensemble_models"]),
            "weights": getattr(engine, "weights", p["ensemble_weights"]),
            "weights_source": getattr(engine, "weights_source", p["ensemble_weights_source"]),
            "weights_optimization_requested": p["ensemble_weights_optimization"],
            "model_errors": getattr(engine, "model_errors", {}),
        }
    if sentiment_model == "meta_learner":
        meta_model_artifact = None
        if getattr(engine, "meta_model_path", None):
            meta_model_artifact = Path(engine.meta_model_path).name
        analysis_meta["meta_learner"] = {
            "model_artifact": meta_model_artifact,
            "base_models": getattr(engine, "base_models", p["meta_learner_models"]),
            "base_models_source": getattr(engine, "base_models_source", None),
            "feature_type": getattr(engine, "feature_type", None),
            "meta_learner_type": getattr(engine, "meta_learner_type", None),
            "model_errors": getattr(engine, "model_errors", {}),
        }
    if sentiment_model == "fuzzy_ensemble":
        analysis_meta["fuzzy"] = {
            "requested_models": getattr(engine, "requested_models", p["fuzzy_models"]),
            "base_models": getattr(engine, "base_models", p["fuzzy_models"]),
            "mf_type": getattr(engine, "mf_type", p["fuzzy_mf_type"]),
            "defuzz_method": getattr(engine, "defuzz_method", p["fuzzy_defuzz_method"]),
            "t_norm": getattr(engine, "t_norm", p["fuzzy_t_norm"]),
            "t_conorm": getattr(engine, "t_conorm", p["fuzzy_t_conorm"]),
            "alpha_cut": getattr(engine, "alpha_cut", p["fuzzy_alpha_cut"]),
            "resolution": getattr(engine, "resolution", p["fuzzy_resolution"]),
            "confidence_threshold": getattr(engine, "confidence_threshold", p["confidence_threshold"]),
            "nf_gate_active": bool(getattr(engine, "_nf_mfs", {})),
            "model_errors": getattr(engine, "model_errors", {}),
        }
    # Expose temperature calibration for any engine that has it
    if hasattr(engine, "temperature"):
        analysis_meta["calibration"] = {
            "temperature": getattr(engine, "temperature", 1.0),
            "applied": getattr(engine, "calibration_applied", False),
        }
    if isinstance(p["model_comparison"], list):
        analysis_meta["model_comparison"] = p["model_comparison"]

    return analysis_meta


def _build_result_payload(video, sentiment_model, processed_comments, filter_stats, analytics, analysis_meta, analysis_id):
    """Steps 8-9: Calculate percentages and assemble the job.result payload
    (the same shape the synchronous endpoint returns directly)."""
    sentiment_data = analytics["sentiment_data"]
    top_positive = analytics["top_positive"]
    top_negative = analytics["top_negative"]
    like_weighted = analytics["like_weighted"]

    total = len(processed_comments)
    sentiment_ratio = {
        'positive_percent': round(sentiment_data['Positive'] / total * 100, 2) if total > 0 else 0,
        'negative_percent': round(sentiment_data['Negative'] / total * 100, 2) if total > 0 else 0,
        'neutral_percent': round(sentiment_data['Neutral'] / total * 100, 2) if total > 0 else 0
    }

    return {
        'msg': 'Analysis complete',
        'video': {
            'id': video.video_id,
            'title': video.title,
            'channel': video.channel_name,
            'view_count': video.view_count,
            'like_count': video.like_count,
            'comment_count': video.comment_count,
            'thumbnail_url': video.thumbnail_url
        },
        'sentiment_data': sentiment_data,
        'sentiment_ratio': sentiment_ratio,
        'total_analyzed': total,
        'filtered': {
            'spam': filter_stats['filtered_spam'],
            'language': filter_stats['filtered_language'],
            'short': filter_stats['filtered_short'],
            # Total comments actually filtered out (spam + language +
            # short) — NOT filter_stats['total'], which is len(comments)
            # fetched *before* filtering. Keeping this consistent with
            # get_user_youtube_analyses()'s 'filtered.total' below, since
            # the frontend displays this field as "Total Filtered".
            'total': (
                filter_stats['filtered_spam']
                + filter_stats['filtered_language']
                + filter_stats['filtered_short']
            ),
        },
        'like_weighted_sentiment': like_weighted[:10],
        'top_words_positive': [{'word': w, 'count': c} for w, c in top_positive[:20]],
        'top_words_negative': [{'word': w, 'count': c} for w, c in top_negative[:20]],
        'confidence_stats': analytics["confidence_stats"],
        'uncertainty_stats': analytics["uncertainty_stats"],
        'sentiment_confidence_intervals': analytics["sentiment_cis"],
        'aspect_sentiment': analytics["aspect_sentiment"],
        'sentiment_timeline': analytics["sentiment_timeline"],
        'analysis_meta': analysis_meta,
        'analysis_id': analysis_id,
        'model_used': sentiment_model.upper()
    }


def _execute_analysis_job(job_id):
    # Atomically claim the job by flipping PENDING -> RUNNING in a single
    # guarded UPDATE. Django-Q2's ORM broker is at-least-once: if a worker is
    # SIGKILLed/OOM-killed after popping this task but before acking it, the
    # cluster redelivers it on restart. Without this guard the redelivery
    # would re-run the whole function and create a *second* YouTubeAnalysis
    # row (and a second result payload) for one request, silently duplicating
    # the analysis in the user's history. Only the caller that wins the
    # PENDING -> RUNNING transition proceeds; a redelivery finds the job
    # already RUNNING/DONE/FAILED, updates 0 rows, and returns. (.update() is
    # a single atomic SQL UPDATE and, unlike .save(), does not fire auto_now,
    # so updated_at is set explicitly to keep the is_stale() clock fresh.)
    claimed = AnalysisJob.objects.filter(
        id=job_id, status=AnalysisJob.STATUS_PENDING
    ).update(status=AnalysisJob.STATUS_RUNNING, updated_at=timezone.now())
    if not claimed:
        logger.warning(
            "Skipping analysis job %s: status is no longer pending "
            "(already claimed, completed, or failed) — most likely a "
            "Django-Q2 task redelivery after a worker restart.",
            job_id,
        )
        return

    job = AnalysisJob.objects.select_related("user").get(id=job_id)
    user = job.user
    p = job.request_params
    video_url = p["video_url"]
    sentiment_model = p["sentiment_model"]

    try:
        fetched = _fetch_comments(job, p, user)
        if fetched is None:
            return
        video_id, video_metadata, comments_raw = fetched

        video = _get_or_update_video(video_id, video_metadata)

        preprocessed = _preprocess_comments(job, p, sentiment_model, comments_raw)
        if preprocessed is None:
            return
        processed_comments, filter_stats, processing_profile = preprocessed

        engine_kwargs = _build_engine_kwargs(sentiment_model, p)
        engine = _run_sentiment_analysis(job, sentiment_model, engine_kwargs, processed_comments, user)
        if engine is None:
            return

        _save_comments(video, processed_comments)

        analytics = _build_analytics(p, processed_comments)
        analysis_meta = _build_analysis_meta(sentiment_model, engine, processing_profile, p, analytics)

        analysis = YouTubeAnalysis.objects.create(
            user=user,
            video=video,
            sentiment_data=analytics["sentiment_data"],
            hour_data=analytics["sentiment_timeline"],
            like_weighted_sentiment=analytics["like_weighted"][:20],
            top_words_positive=[{'word': w, 'count': c} for w, c in analytics["top_positive"]],
            top_words_negative=[{'word': w, 'count': c} for w, c in analytics["top_negative"]],
            total_comments_analyzed=len(processed_comments),
            filtered_spam_count=filter_stats['filtered_spam'],
            filtered_language_count=filter_stats['filtered_language'],
            filtered_short_count=filter_stats['filtered_short'],
            analysis_model=sentiment_model.upper(),
            analysis_meta=analysis_meta,
        )

        logger.debug(
            "Analysis complete for user=%s video=%s model=%s",
            user.id,
            video.video_id,
            sentiment_model,
        )

        job.status = AnalysisJob.STATUS_DONE
        job.result = _build_result_payload(
            video, sentiment_model, processed_comments, filter_stats,
            analytics, analysis_meta, analysis.id,
        )
        job.save(update_fields=["status", "result", "updated_at"])

    except Exception:
        logger.exception(
            "Analysis failed for user=%s video_url=%s",
            getattr(user, "id", None),
            video_url,
        )
        _fail_job(job, "Analysis failed due to an internal error. Please try again later.", 500)


# @api_view wraps the function in a dynamically-created APIView subclass and
# returns `.as_view()`; DRF's ScopedRateThrottle reads `throttle_scope` off the
# view *class* (via `view.cls`, set by APIView.as_view()), not off the plain
# function, so this must be assigned here rather than inside the function body.
analyze_youtube_video.cls.throttle_scope = "analyze"


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_analysis_job_status(request, job_id):
    """
    Poll the status of a background `youtube/analyze/` job (see
    app/models.py::AnalysisJob). Only meaningful when ANALYSIS_RUN_SYNC is
    False — the synchronous path (tests, or ANALYSIS_RUN_SYNC=true) never
    returns a job_id to poll in the first place.
    """
    try:
        job = AnalysisJob.objects.get(id=job_id, user=request.user)
    except AnalysisJob.DoesNotExist:
        return Response({"msg": "No such analysis job"}, status=404)

    if job.is_stale():
        # The Django-Q2 worker almost certainly died without ever
        # recording a result (process restart/crash/OOM) — resolve it now
        # instead of leaving the client to poll a job that will never move
        # out of "running"/"pending".
        job.mark_stale_failed()

    if job.status == AnalysisJob.STATUS_DONE:
        return Response({"status": job.status, **job.result})
    if job.status == AnalysisJob.STATUS_FAILED:
        return Response(
            {"status": job.status, "msg": job.error_message},
            status=job.error_status or 500,
        )
    return Response({"status": job.status})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_youtube_analysis(request, video_id):
    try:
        analysis = YouTubeAnalysis.objects.filter(
            user=request.user,
            video__video_id=video_id,
        ).order_by('-fetched_date').first()

        if not analysis:
            return Response(
                {"msg": "No analysis found for this video"},
                status=404
            )

        total = analysis.total_comments_analyzed
        sentiment_ratio = {
            'positive_percent': round(analysis.sentiment_data['Positive'] / total * 100, 2) if total > 0 else 0,
            'negative_percent': round(analysis.sentiment_data['Negative'] / total * 100, 2) if total > 0 else 0,
            'neutral_percent': round(analysis.sentiment_data['Neutral'] / total * 100, 2) if total > 0 else 0
        }

        return Response({
            'data': {
                'video': {
                    'id': analysis.video.video_id,
                    'title': analysis.video.title,
                    'channel': analysis.video.channel_name,
                    'view_count': analysis.video.view_count,
                    'like_count': analysis.video.like_count,
                    'thumbnail_url': analysis.video.thumbnail_url
                },
                'sentiment_data': analysis.sentiment_data,
                'sentiment_ratio': sentiment_ratio,
                'like_weighted_sentiment': analysis.like_weighted_sentiment,
                'top_words_positive': analysis.top_words_positive,
                'top_words_negative': analysis.top_words_negative,
                'total_comments': analysis.total_comments_analyzed,
                'sentiment_timeline': analysis.hour_data,
                'filtered': {
                    'spam': analysis.filtered_spam_count,
                    'language': analysis.filtered_language_count,
                    'short': analysis.filtered_short_count,
                    'total': (
                        analysis.filtered_spam_count
                        + analysis.filtered_language_count
                        + analysis.filtered_short_count
                    ),
                },
                'confidence_stats': (analysis.analysis_meta or {}).get('confidence_stats'),
                'uncertainty_stats': (analysis.analysis_meta or {}).get('uncertainty_stats'),
                'calibration': (analysis.analysis_meta or {}).get('calibration'),
                'sentiment_confidence_intervals': (analysis.analysis_meta or {}).get(
                    'sentiment_confidence_intervals'
                ),
                'aspect_sentiment': (analysis.analysis_meta or {}).get('aspect_sentiment'),
                'analysis_meta': analysis.analysis_meta,
                'model_used': analysis.analysis_model,
                'fetched_date': analysis.fetched_date
            }
        })

    except Exception:
        logger.exception(
            "get_youtube_analysis failed for user=%s video_id=%s",
            getattr(request.user, "id", None),
            video_id,
        )
        return Response(
            {"msg": "Failed to retrieve analysis due to an internal error."},
            status=500,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_user_youtube_analyses(request):
    try:
        analyses = YouTubeAnalysis.objects.filter(
            user=request.user
        ).select_related('video').order_by('-fetched_date')[:20]

        data = []
        for analysis in analyses:
            total = analysis.total_comments_analyzed
            data.append({
                'id': analysis.id,
                'video': {
                    'id': analysis.video.video_id,
                    'title': analysis.video.title,
                    'channel': analysis.video.channel_name,
                    'channel_id': analysis.video.channel_id,
                    'view_count': analysis.video.view_count,
                    'like_count': analysis.video.like_count,
                    'thumbnail_url': analysis.video.thumbnail_url
                },
                'sentiment_data': analysis.sentiment_data,
                'total_comments_analyzed': total,
                'positive_percent': round(analysis.sentiment_data['Positive'] / total * 100, 2) if total > 0 else 0,
                'negative_percent': round(analysis.sentiment_data['Negative'] / total * 100, 2) if total > 0 else 0,
                'like_weighted_sentiment': analysis.like_weighted_sentiment,
                'top_words_positive': analysis.top_words_positive,
                'top_words_negative': analysis.top_words_negative,
                'filtered': {
                    'spam': analysis.filtered_spam_count,
                    'language': analysis.filtered_language_count,
                    'short': analysis.filtered_short_count,
                    'total': analysis.filtered_spam_count + analysis.filtered_language_count + analysis.filtered_short_count
                },
                'analysis_model': analysis.analysis_model,
                'fetched_date': analysis.fetched_date,
                'confidence_stats': (analysis.analysis_meta or {}).get('confidence_stats'),
                'uncertainty_stats': (analysis.analysis_meta or {}).get('uncertainty_stats'),
                'calibration': (analysis.analysis_meta or {}).get('calibration'),
            })

        return Response({'data': data})

    except Exception:
        logger.exception(
            "get_user_youtube_analyses failed for user=%s",
            getattr(request.user, "id", None),
        )
        return Response(
            {"msg": "Failed to retrieve analyses due to an internal error."},
            status=500,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
@throttle_classes([ScopedRateThrottle])
def search_youtube_videos(request):
    """
    Lets the Search page offer a "search YouTube" picker instead of requiring
    the user to already have a video URL copied — the user types a keyword
    query and picks from a results list (title/channel/thumbnail), which then
    fills in the video_url field for the existing analyze flow unchanged.

    Uses the official Data API only (no scraper fallback): search().list
    costs a flat 100 quota units per call regardless of maxResults, which is
    why this is throttled separately (see 'search' in DEFAULT_THROTTLE_RATES)
    and why max_results is capped well below analyze's comment-count limits.
    """
    query = (request.GET.get("q") or "").strip()
    if not query:
        return Response({"msg": "A search query (q) is required."}, status=400)
    if len(query) > 100:
        return Response({"msg": "Search query is too long (max 100 characters)."}, status=400)

    raw_max_results = request.GET.get("max_results")
    try:
        max_results = _parse_bounded_int(
            raw_max_results, 8, minimum=1, maximum=15, name="max_results"
        )
    except _InvalidParam as exc:
        return Response({"msg": str(exc)}, status=400)

    try:
        fetcher = YouTubeFetcher()
    except ValueError as e:
        # Local configuration error (missing YOUTUBE_API_KEY) — safe to
        # surface verbatim, it never touches the network or embeds request
        # details. Search has no scraper-mode fallback, unlike analyze.
        return Response({"msg": str(e)}, status=502)

    try:
        results = fetcher.search_videos(query, max_results=max_results)
    except HttpError as e:
        # Never surface str(e)/repr(e) here: HttpError.__str__ always embeds
        # the full request URL (self.uri), which for this client includes
        # `key=<YOUTUBE_API_KEY>` as a query parameter.
        status_code = getattr(e.resp, "status", None)
        try:
            error_details = json.loads(e.content).get('error', {})
            reason = error_details.get('errors', [{}])[0].get('reason')
            message = error_details.get('message') or (
                f"YouTube API request failed with status {status_code}."
            )
            if reason == 'quotaExceeded':
                return Response(
                    {"msg": "YouTube API daily quota exceeded. Please try again tomorrow."},
                    status=429,
                )
            if reason == 'developerKeyInvalid':
                return Response(
                    {"msg": "The provided YOUTUBE_API_KEY is invalid. Please check your .env file."},
                    status=401,
                )
            return Response({"msg": f"A YouTube API error occurred: {message}"}, status=502)
        except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
            logger.warning("Unparseable YouTube API error response (status=%s)", status_code)
            return Response(
                {"msg": f"A YouTube API error occurred (status {status_code})."},
                status=502,
            )
    except Exception:
        logger.exception(
            "Unexpected error searching YouTube for user=%s query=%r",
            getattr(request.user, "id", None),
            query,
        )
        return Response(
            {"msg": "An unexpected error occurred while searching YouTube."},
            status=502,
        )

    return Response({"data": results})


# See the throttle_scope comment above analyze_youtube_video.cls.throttle_scope.
search_youtube_videos.cls.throttle_scope = "search"


# Health check endpoint — unauthenticated so load balancers / uptime
# monitors can poll it, and backed by real checks rather than a hardcoded
# string: database reachability and presence of the model artifacts the
# default (logreg) sentiment engine needs to serve a request.
#
# Note: the pinned runtime manifest's "path" field for model artifacts
# (logreg_model, etc.) is relative to the runtime-artifact directory, not to
# `Config.MODELS_DIR` where the actual model files live — only the JSON
# research artifacts resolve correctly via `resolve_runtime_artifact_path`
# (see the docstring on `_verify_model_artifact_hash_uncached`). So this
# checks the same default paths `LogRegSentimentEngine` actually loads from.
@api_view(["GET"])
@permission_classes((AllowAny,))
@throttle_classes([])
def index(request):
    # Unauthenticated (AllowAny) so load balancers / uptime monitors can poll
    # it — so the response must never include exception text (can embed
    # connection details) or server-side filesystem paths. Real detail goes
    # to the server log only.
    #
    # @throttle_classes([]) exempts it from the default AnonRateThrottle
    # (20/minute): a load balancer or uptime monitor polling faster than
    # every 3s — or several sharing one proxy IP — would otherwise get 429s
    # and mark the backend down. A health check must not be rate-limited.
    checks = {}

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        checks["database"] = "ok"
    except DjangoDBError:
        logger.exception("Health check: database connectivity failed")
        checks["database"] = "error"

    missing_artifacts = [
        path
        for path in (
            Config.MODELS_DIR / "logreg" / "model.sav",
            Config.MODELS_DIR / "logreg" / "tfidfVectorizer.pickle",
        )
        if not path.exists()
    ]
    if missing_artifacts:
        logger.error(
            "Health check: missing model artifacts: %s",
            ", ".join(str(path) for path in missing_artifacts),
        )
    checks["model_artifacts"] = "ok" if not missing_artifacts else "error"

    healthy = all(value == "ok" for value in checks.values())
    return JsonResponse(
        {
            "status": "ok" if healthy else "unhealthy",
            "checks": checks,
        },
        status=200 if healthy else 503,
    )
