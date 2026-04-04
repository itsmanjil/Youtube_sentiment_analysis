
import json
import os
import logging
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

from googleapiclient.errors import HttpError

from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import *
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
    "xlm-v": "xlm_v",
    "mdeberta-v3": "mdeberta_v3",
}

_TRANSFORMER_MODELS = {
    "bert",
    "transformer",
    "roberta",
    "modernbert",
    "deberta_v3",
    "xlm_v",
    "mdeberta_v3",
}


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



@api_view(["POST"])
@permission_classes([IsAuthenticated])
def analyze_youtube_video(request):
    user = request.user

    # Extract parameters
    video_url = request.data.get("video_url")
    max_comments = request.data.get("max_comments", 200)
    use_api = request.data.get("use_api", True)
    emoji_mode = request.data.get("emoji_mode", "convert")
    filter_spam = request.data.get("filter_spam", True)
    filter_language = request.data.get("filter_language", True)
    sentiment_model = _normalize_sentiment_model(request.data.get("sentiment_model", "logreg"))
    calibration_profile = str(request.data.get("calibration_profile", "auto") or "auto")
    bootstrap_samples = int(request.data.get("bootstrap_samples", 500))
    random_seed = int(request.data.get("random_seed", 42))
    aspect_top_n = int(request.data.get("aspect_top_n", 12))
    aspect_min_freq = int(request.data.get("aspect_min_freq", 3))
    confidence_threshold = float(request.data.get("confidence_threshold", 0.6))
    ensemble_models = _coerce_model_list(request.data.get("ensemble_models"))
    ensemble_weights_optimization = request.data.get("ensemble_weights_optimization")
    ensemble_weights_input = request.data.get("ensemble_weights")
    meta_learner_path = request.data.get("meta_learner_path")
    meta_learner_models = _coerce_model_list(request.data.get("meta_learner_models"))
    fuzzy_models = _coerce_model_list(request.data.get("fuzzy_models"))
    fuzzy_mf_type = request.data.get("fuzzy_mf_type")
    fuzzy_defuzz_method = request.data.get("fuzzy_defuzz_method")
    fuzzy_t_norm = request.data.get("fuzzy_t_norm")
    fuzzy_t_conorm = request.data.get("fuzzy_t_conorm")
    fuzzy_alpha_cut = request.data.get("fuzzy_alpha_cut")
    fuzzy_resolution = request.data.get("fuzzy_resolution")
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

    try:
        # Step 1: Fetch comments
        logger.debug("Fetching comments from %s", video_url)
        if use_api:
            try:
                fetcher = YouTubeFetcher()
                video_id = fetcher.extract_video_id(video_url)
                video_metadata = fetcher.fetch_video_metadata(video_id)
                comments_raw = fetcher.fetch_comments(video_url, max_results=max_comments)
            except HttpError as e:
                try:
                    error_details = json.loads(e.content).get('error', {})
                    reason = error_details.get('errors', [{}])[0].get('reason')
                    message = error_details.get('message', str(e))

                    if reason == 'quotaExceeded':
                        return Response({"msg": "YouTube API daily quota exceeded. Please try again tomorrow or use scraper mode (use_api: false)."}, status=429)
                    elif reason == 'developerKeyInvalid':
                        return Response({"msg": "The provided YOUTUBE_API_KEY is invalid. Please check your .env file and ensure it is correct."}, status=401)
                    elif reason == 'commentsDisabled':
                        return Response({"msg": "Comments are disabled for this video."}, status=403)
                    elif e.resp.status == 404:
                        return Response({"msg": "Video not found. Please check the URL."}, status=404)
                    else:
                        return Response({"msg": f"A YouTube API error occurred: {message}"}, status=502)
                except (json.JSONDecodeError, KeyError, IndexError):
                     return Response({"msg": f"An unhandled YouTube API error occurred: {str(e)}"}, status=502)
            except Exception as e:
                return Response(
                    {"msg": f"An unexpected error occurred with the YouTube API client: {str(e)}. Please ensure your YOUTUBE_API_KEY is correctly set in the .env file."},
                    status=502
                )
        else:
            try:
                scraper = YouTubeScraper()
                video_id = scraper.extract_video_id(video_url)
                video_metadata = scraper.fetch_video_metadata(video_id)
                comments_raw = scraper.fetch_comments(video_url, max_results=max_comments)
            except Exception as e:
                return Response(
                    {"msg": f"Scraper error: {str(e)}"},
                    status=502
                )

        if not comments_raw:
            return Response(
                {"msg": "No comments found for this video"},
                status=404
            )

        logger.debug("Fetched %s raw comments", len(comments_raw))

        # Step 2: Save or update video metadata
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

        # Step 3: Preprocess comments
        logger.debug("Preprocessing comments")
        preprocessor = YouTubePreprocessor()
        processing_profile = _get_processing_profile(sentiment_model)
        processed_comments, filter_stats = preprocessor.batch_preprocess(
            comments_raw,
            profile=processing_profile,
            emoji_mode=emoji_mode,
            check_spam=filter_spam,
            check_lang=filter_language
        )

        if not processed_comments:
            return Response(
                {"msg": "All comments were filtered out. Try different filter settings."},
                status=400
            )

        logger.debug("Processed %s comments after filtering", len(processed_comments))

        # Step 4: Sentiment Analysis
        logger.debug("Running sentiment analysis using %s", sentiment_model)
        engine_kwargs = {}
        if sentiment_model == "ensemble":
            engine_kwargs = {
                "base_models": ensemble_models,
                "weights": ensemble_weights,
                "weights_optimization": ensemble_weights_optimization,
            }
        elif sentiment_model == "meta_learner":
            if meta_learner_path:
                engine_kwargs["meta_model_path"] = meta_learner_path
            if meta_learner_models:
                engine_kwargs["base_models"] = meta_learner_models
        elif sentiment_model == "fuzzy_ensemble":
            alpha_cut = 0.0
            if fuzzy_alpha_cut is not None:
                try:
                    alpha_cut = float(fuzzy_alpha_cut)
                except (TypeError, ValueError):
                    alpha_cut = 0.0
            resolution = 100
            if fuzzy_resolution is not None:
                try:
                    resolution = int(fuzzy_resolution)
                except (TypeError, ValueError):
                    resolution = 100

            engine_kwargs = {
                "base_models": fuzzy_models,
                "mf_type": fuzzy_mf_type or "gaussian",
                "defuzz_method": fuzzy_defuzz_method or "centroid",
                "t_norm": fuzzy_t_norm or "min",
                "t_conorm": fuzzy_t_conorm or "max",
                "alpha_cut": alpha_cut,
                "resolution": resolution,
                "confidence_threshold": confidence_threshold,
            }
        elif _is_transformer_model(sentiment_model):
            engine_kwargs = {
                "calibration_profile": calibration_profile,
            }
        try:
            engine = get_sentiment_engine(sentiment_model, **engine_kwargs)
        except (ValueError, RuntimeError, ImportError, FileNotFoundError) as exc:
            return Response({"msg": str(exc)}, status=400)

        for item in processed_comments:
            result = coerce_sentiment_result(
                engine.analyze(item['processed_text']),
                sentiment_model,
            )
            item['sentiment'] = result.label
            item['sentiment_score'] = result.score
            item['sentiment_probs'] = result.probs
            item['confidence'] = confidence_from_probs(result.probs)

        # Step 5: Save comments to database
        logger.debug("Saving processed comments to database")
        for item in processed_comments:
            try:
                YouTubeComment.objects.update_or_create(
                    comment_id=item.get('comment_id', ''),
                    defaults={
                        'video': video,
                        'text': item['text'],
                        'author': item['author'],
                        'likes': item['likes'],
                        'published_at': item['published_at'],
                        'is_reply': item['is_reply'],
                        'sentiment': item['sentiment'],
                        'sentiment_score': item['sentiment_score'],
                        'is_spam': False,
                        'language': item['metadata']['language']
                    }
                )
            except Exception as e:
                logger.warning(
                    "Failed to save comment %s for video %s: %s",
                    item.get('comment_id', ''),
                    video.video_id,
                    e,
                )
                continue

        # Step 6: Generate analytics
        logger.debug("Generating analysis aggregates")
        sentiments = [item['sentiment'] for item in processed_comments]
        sentiment_counts = Counter(sentiments)
        confidences = [item.get('confidence', 0.0) for item in processed_comments]
        confidence_stats = aggregate_confidence_stats(
            confidences,
            threshold=confidence_threshold,
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
            n_boot=bootstrap_samples,
            alpha=0.05,
            seed=random_seed,
        )
        aspect_sentiment = extract_aspect_sentiment(
            processed_comments,
            top_n=aspect_top_n,
            min_freq=aspect_min_freq,
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

        analysis_meta = {
            "model_family": _get_model_family(sentiment_model),
            "model_artifact": getattr(engine, "model_artifact", None),
            "preprocessing_profile": processing_profile,
            "runtime_artifacts": get_runtime_artifact_metadata(),
            "confidence_stats": confidence_stats,
            "uncertainty_stats": uncertainty_stats,
            "sentiment_confidence_intervals": sentiment_cis,
            "aspect_sentiment": aspect_sentiment,
            "bootstrap_samples": bootstrap_samples,
            "random_seed": random_seed,
        }
        if _is_transformer_model(sentiment_model):
            analysis_meta["transformer"] = {
                "preset": getattr(engine, "model_preset", None),
                "source": getattr(engine, "model_source", None),
                "artifact": getattr(engine, "model_artifact", None),
                "max_length": getattr(engine, "max_length", None),
                "device": str(getattr(engine, "device", "")),
                "calibration_profile": getattr(engine, "calibration_profile", calibration_profile),
                "calibration_applied": getattr(engine, "calibration_applied", False),
                "temperature": getattr(engine, "temperature", None),
                "temperature_artifact_path": getattr(engine, "temperature_artifact_path", None),
            }
        if sentiment_model == "ensemble":
            analysis_meta["ensemble"] = {
                "models": getattr(engine, "requested_models", ensemble_models),
                "models_used": getattr(engine, "base_models", ensemble_models),
                "weights": getattr(engine, "weights", ensemble_weights),
                "weights_source": getattr(engine, "weights_source", ensemble_weights_source),
                "weights_optimization_requested": ensemble_weights_optimization,
                "model_errors": getattr(engine, "model_errors", {}),
            }
        if sentiment_model == "meta_learner":
            meta_model_artifact = None
            if getattr(engine, "meta_model_path", None):
                meta_model_artifact = Path(engine.meta_model_path).name
            analysis_meta["meta_learner"] = {
                "model_artifact": meta_model_artifact,
                "base_models": getattr(engine, "base_models", meta_learner_models),
                "base_models_source": getattr(engine, "base_models_source", None),
                "feature_type": getattr(engine, "feature_type", None),
                "meta_learner_type": getattr(engine, "meta_learner_type", None),
                "model_errors": getattr(engine, "model_errors", {}),
            }
        if sentiment_model == "fuzzy_ensemble":
            analysis_meta["fuzzy"] = {
                "requested_models": getattr(engine, "requested_models", fuzzy_models),
                "base_models": getattr(engine, "base_models", fuzzy_models),
                "mf_type": getattr(engine, "mf_type", fuzzy_mf_type),
                "defuzz_method": getattr(engine, "defuzz_method", fuzzy_defuzz_method),
                "t_norm": getattr(engine, "t_norm", fuzzy_t_norm),
                "t_conorm": getattr(engine, "t_conorm", fuzzy_t_conorm),
                "alpha_cut": getattr(engine, "alpha_cut", fuzzy_alpha_cut),
                "resolution": getattr(engine, "resolution", fuzzy_resolution),
                "confidence_threshold": getattr(engine, "confidence_threshold", confidence_threshold),
                "nf_gate_active": bool(getattr(engine, "_nf_mfs", {})),
                "model_errors": getattr(engine, "model_errors", {}),
            }
        # Expose temperature calibration for any engine that has it
        if hasattr(engine, "temperature"):
            analysis_meta["calibration"] = {
                "temperature": getattr(engine, "temperature", 1.0),
                "applied": getattr(engine, "calibration_applied", False),
            }
        if isinstance(model_comparison, list):
            analysis_meta["model_comparison"] = model_comparison

        # Step 7: Save analysis
        analysis = YouTubeAnalysis.objects.create(
            user=user,
            video=video,
            sentiment_data=sentiment_data,
            hour_data=sentiment_timeline,
            like_weighted_sentiment=like_weighted[:20],
            top_words_positive=[{'word': w, 'count': c} for w, c in top_positive],
            top_words_negative=[{'word': w, 'count': c} for w, c in top_negative],
            total_comments_analyzed=len(processed_comments),
            filtered_spam_count=filter_stats['filtered_spam'],
            filtered_language_count=filter_stats['filtered_language'],
            filtered_short_count=filter_stats['filtered_short'],
            analysis_model=sentiment_model.upper(),
            analysis_meta=analysis_meta,
        )

        # Step 8: Calculate percentages
        total = len(processed_comments)
        sentiment_ratio = {
            'positive_percent': round(sentiment_data['Positive'] / total * 100, 2) if total > 0 else 0,
            'negative_percent': round(sentiment_data['Negative'] / total * 100, 2) if total > 0 else 0,
            'neutral_percent': round(sentiment_data['Neutral'] / total * 100, 2) if total > 0 else 0
        }

        logger.debug(
            "Analysis complete for user=%s video=%s model=%s",
            user.id,
            video.video_id,
            sentiment_model,
        )

        # Step 9: Return response
        return Response({
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
            'total_analyzed': len(processed_comments),
            'filtered': {
                'spam': filter_stats['filtered_spam'],
                'language': filter_stats['filtered_language'],
                'short': filter_stats['filtered_short'],
                'total': filter_stats['total']
            },
            'like_weighted_sentiment': like_weighted[:10],
            'top_words_positive': [{'word': w, 'count': c} for w, c in top_positive[:20]],
            'top_words_negative': [{'word': w, 'count': c} for w, c in top_negative[:20]],
            'confidence_stats': confidence_stats,
            'uncertainty_stats': uncertainty_stats,
            'sentiment_confidence_intervals': sentiment_cis,
            'aspect_sentiment': aspect_sentiment,
            'sentiment_timeline': sentiment_timeline,
            'analysis_meta': analysis_meta,
            'analysis_id': analysis.id,
            'model_used': sentiment_model.upper()
        })

    except Exception as e:
        logger.exception(
            "Analysis failed for user=%s video_url=%s",
            getattr(user, "id", None),
            video_url,
        )
        return Response(
            {"msg": f"Analysis failed: {str(e)}"},
            status=500
        )


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
                    'short': analysis.filtered_short_count
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

    except Exception as e:
        return Response({"msg": str(e)}, status=500)


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

    except Exception as e:
        return Response({"msg": str(e)}, status=500)


# Health check endpoint
@api_view(["GET"])
@permission_classes((IsAuthenticated,))
def index(request):
    return JsonResponse({"data": "YouTube Sentiment Analysis API - v2.0"})


# Test endpoint
@api_view(["GET"])
def test_endpoint(request):
    return JsonResponse({"status": "Server is working", "message": "Test successful"})
