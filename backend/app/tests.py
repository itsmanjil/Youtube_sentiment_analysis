import json
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

import numpy as np
from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from googleapiclient.errors import HttpError

from users.models import NewUser
from core.settings_utils import DEV_SECRET_KEY, resolve_runtime_settings
from .analysis_utils import (
    aggregate_confidence_stats,
    bootstrap_confidence_intervals,
    normalize_probs,
)
from .models import YouTubeVideo, YouTubeAnalysis, YouTubeComment
from .youtube_preprocessor import YouTubePreprocessor
from src.sentiment import SentimentResult
from research.transformers.model_registry import get_encoder_spec
from research.transformers.train_encoder import (
    load_split_metadata,
    resolve_text_column,
    summarize_split_provenance,
)
from src.utils.calibration import (
    apply_temperature_to_logits,
    load_temperature_artifact,
    save_temperature_artifact,
)

# Mock data for YouTube Fetcher/Scraper
MOCK_VIDEO_METADATA = {
    'title': 'Test Video Title',
    'description': 'A description for the test video.',
    'channel': 'Test Channel',
    'channel_id': 'UC-test-channel',
    'published_at': '2026-01-11T00:00:00Z',
    'view_count': 1000,
    'like_count': 100,
    'comment_count': 10,
    'thumbnail_url': 'https://test.com/thumb.jpg'
}

MOCK_COMMENTS_RAW = [
    {'comment_id': 'c1', 'text': 'This is a great video!', 'author': 'user1', 'likes': 10, 'published_at': '2026-01-11T01:00:00Z', 'is_reply': False},
    {'comment_id': 'c2', 'text': 'I did not like this.', 'author': 'user2', 'likes': 2, 'published_at': '2026-01-11T02:00:00Z', 'is_reply': False},
    {'comment_id': 'c3', 'text': 'Just a neutral comment.', 'author': 'user3', 'likes': 5, 'published_at': '2026-01-11T03:00:00Z', 'is_reply': False},
    {'comment_id': 'c4', 'text': 'check out my channel', 'author': 'spammer', 'likes': 0, 'published_at': '2026-01-11T04:00:00Z', 'is_reply': False}, # This should be filtered as spam
]

# Mock for sentiment engine
class MockSentimentEngine:
    def analyze(self, text):
        if "great" in text:
            label = "Positive"
            score = 0.8
        elif "not like" in text or "dislike" in text:
            label = "Negative"
            score = -0.5
        else:
            label = "Neutral"
            score = 0.0
        return SentimentResult(
            label=label,
            score=score,
            probs=normalize_probs({label: 1.0}),
            model="mock",
            raw={"compound": score},
        )

class YouTubeAnalysisAPITests(APITestCase):
    def setUp(self):
        # Create a test user
        self.user = NewUser.objects.create_user(
            email='test@example.com',
            user_name='testuser',
            first_name='Test',
            last_name='User',
            password='testpassword123',
        )
        self.client.force_authenticate(user=self.user)

        # URLs
        self.analyze_url = reverse('app:youtube_analyze')

    @patch('app.views.YouTubeFetcher')
    @patch('app.views.get_sentiment_engine')
    def test_analyze_video_success_api_mode(self, mock_get_engine, mock_fetcher):
        # Setup mocks
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.extract_video_id.return_value = 'HLUamwXQ218'
        mock_fetcher_instance.fetch_video_metadata.return_value = MOCK_VIDEO_METADATA
        mock_fetcher_instance.fetch_comments.return_value = MOCK_COMMENTS_RAW

        mock_get_engine.return_value = MockSentimentEngine()

        data = {
            "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
            "max_comments": 100,
            "use_api": True,
            "filter_spam": True,
            "filter_language": False,
        }

        response = self.client.post(self.analyze_url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['msg'], 'Analysis complete')
        self.assertEqual(YouTubeVideo.objects.count(), 1)
        self.assertEqual(YouTubeAnalysis.objects.count(), 1)
        # 3 comments should be saved (1 spam comment is filtered)
        self.assertEqual(YouTubeComment.objects.count(), 3)

        analysis = YouTubeAnalysis.objects.first()
        self.assertEqual(analysis.user, self.user)
        self.assertEqual(analysis.video.video_id, 'HLUamwXQ218')
        self.assertEqual(analysis.total_comments_analyzed, 3)
        self.assertEqual(analysis.filtered_spam_count, 1)
        self.assertEqual(analysis.sentiment_data['Positive'], 1)
        # MockSentimentEngine marks "I did not like this." as Negative.
        self.assertEqual(analysis.sentiment_data['Negative'], 1)
        self.assertEqual(analysis.sentiment_data['Neutral'], 1)

    @patch('app.views.YouTubeFetcher')
    @patch('app.views.get_sentiment_engine')
    def test_analyze_video_uses_transformer_preprocessing_for_encoder_models(self, mock_get_engine, mock_fetcher):
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.extract_video_id.return_value = 'HLUamwXQ218'
        mock_fetcher_instance.fetch_video_metadata.return_value = MOCK_VIDEO_METADATA
        mock_fetcher_instance.fetch_comments.return_value = MOCK_COMMENTS_RAW[:3]

        mock_engine = MockSentimentEngine()
        mock_engine.model_preset = "modernbert"
        mock_engine.model_source = "answerdotai/ModernBERT-base"
        mock_engine.model_artifact = "ModernBERT-base"
        mock_engine.calibration_applied = False
        mock_engine.calibration_profile = "auto"
        mock_engine.temperature = 1.0
        mock_engine.temperature_artifact_path = None
        mock_engine.max_length = 128
        mock_engine.device = "cpu"
        mock_get_engine.return_value = mock_engine

        data = {
            "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
            "sentiment_model": "modernbert",
            "filter_spam": False,
            "filter_language": False,
        }

        response = self.client.post(self.analyze_url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["model_used"], "MODERNBERT")
        self.assertEqual(response.data["analysis_meta"]["model_family"], "transformer")
        self.assertEqual(response.data["analysis_meta"]["preprocessing_profile"], "transformer")
        self.assertEqual(response.data["analysis_meta"]["transformer"]["preset"], "modernbert")
        self.assertEqual(response.data["analysis_meta"]["transformer"]["calibration_profile"], "auto")
        mock_get_engine.assert_called_with("modernbert", calibration_profile="auto")

    def test_analyze_video_missing_url(self):
        data = {"max_comments": 100}
        response = self.client.post(self.analyze_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data['msg'], 'video_url is required')

    def test_analyze_video_rejects_path_based_ensemble_weights(self):
        data = {
            "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
            "sentiment_model": "ensemble",
            "ensemble_weights": "backend/models/pso_ensemble_weights.json",
        }

        response = self.client.post(self.analyze_url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("inline JSON weights", response.data["msg"])

    def test_analyze_video_rejects_meta_learner_path_override(self):
        data = {
            "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
            "sentiment_model": "meta_learner",
            "meta_learner_path": "backend/models/meta_learner.pkl",
        }

        response = self.client.post(self.analyze_url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("meta_learner_path overrides are not supported", response.data["msg"])

    @patch('app.views.YouTubeFetcher')
    def test_analyze_video_api_error_quota(self, mock_fetcher):
        mock_error_content = b'{"error": {"errors": [{"reason": "quotaExceeded"}], "message": "Quota Exceeded"}}'
        mock_resp = MagicMock(status=403)
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.fetch_comments.side_effect = HttpError(resp=mock_resp, content=mock_error_content)

        data = {"video_url": "https://www.youtube.com/watch?v=somevideo"}
        response = self.client.post(self.analyze_url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_429_TOO_MANY_REQUESTS)
        self.assertIn("quota exceeded", response.data['msg'])

    def test_get_user_analyses(self):
        video = YouTubeVideo.objects.create(
            video_id='v1',
            title='Video 1',
            channel_name='Channel 1',
            published_at='2026-01-01T00:00:00Z',
        )
        YouTubeAnalysis.objects.create(
            user=self.user,
            video=video,
            sentiment_data={'Positive': 1, 'Neutral': 0, 'Negative': 0},
            total_comments_analyzed=1,
        )
        other_user = NewUser.objects.create_user(
            email='other@test.com',
            user_name='otheruser',
            first_name='Other',
            last_name='User',
            password='password',
        )
        YouTubeAnalysis.objects.create(
            user=other_user,
            video=video,
            sentiment_data={'Positive': 3, 'Neutral': 0, 'Negative': 0},
            total_comments_analyzed=3,
        )

        url = reverse('app:get_user_analyses')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']), 1)

    def test_get_single_analysis(self):
        video = YouTubeVideo.objects.create(
            video_id='v1',
            title='Video 1',
            channel_name='Channel 1',
            published_at='2026-01-01T00:00:00Z',
        )
        YouTubeAnalysis.objects.create(
            user=self.user,
            video=video,
            sentiment_data={'Positive': 10, 'Neutral': 0, 'Negative': 0},
            total_comments_analyzed=10,
            analysis_model='LOGREG',
        )

        url = reverse('app:get_youtube_analysis', kwargs={'video_id': 'v1'})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['video']['id'], 'v1')
        self.assertEqual(response.data['data']['model_used'], 'LOGREG')

    def test_get_single_analysis_is_scoped_to_authenticated_user(self):
        video = YouTubeVideo.objects.create(
            video_id='v1',
            title='Video 1',
            channel_name='Channel 1',
            published_at='2026-01-01T00:00:00Z',
        )
        YouTubeAnalysis.objects.create(
            user=self.user,
            video=video,
            sentiment_data={'Positive': 1, 'Neutral': 0, 'Negative': 0},
            total_comments_analyzed=1,
            analysis_model='LOGREG',
        )
        other_user = NewUser.objects.create_user(
            email='other2@test.com',
            user_name='otheruser2',
            first_name='Other',
            last_name='User',
            password='password',
        )
        YouTubeAnalysis.objects.create(
            user=other_user,
            video=video,
            sentiment_data={'Positive': 0, 'Neutral': 0, 'Negative': 5},
            total_comments_analyzed=5,
            analysis_model='SVM',
        )

        url = reverse('app:get_youtube_analysis', kwargs={'video_id': 'v1'})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['model_used'], 'LOGREG')
        self.assertEqual(response.data['data']['sentiment_data']['Positive'], 1)
        self.assertEqual(response.data['data']['sentiment_data']['Negative'], 0)

    def test_get_single_analysis_not_found(self):
        url = reverse('app:get_youtube_analysis', kwargs={'video_id': 'nonexistent'})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_health_check_endpoint(self):
        url = reverse('app:youtube_health_check')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()['data'], 'YouTube Sentiment Analysis API - v2.0')


class AnalysisUtilsTests(APITestCase):
    def test_confidence_stats_and_intervals(self):
        confidences = [0.2, 0.5, 0.9]
        stats = aggregate_confidence_stats(confidences, threshold=0.6)
        self.assertIn("mean", stats)
        self.assertIn("low_confidence_ratio", stats)

        labels = ["Positive"] * 5 + ["Negative"] * 5
        intervals = bootstrap_confidence_intervals(labels, n_boot=10, seed=1)
        self.assertIn("Positive", intervals)
        self.assertLessEqual(
            intervals["Positive"]["lower"],
            intervals["Positive"]["upper"],
        )


class YouTubePreprocessorTests(SimpleTestCase):
    def test_profiles_preserve_transformer_cues_but_clean_classical_text(self):
        preprocessor = YouTubePreprocessor()
        text = "WOW!!! This video is sooo good 😍 #Amazing @channel 01:23"

        classical, classical_meta = preprocessor.preprocess_youtube_comment(
            text,
            check_spam=False,
            check_lang=False,
            profile="classical",
        )
        transformer, transformer_meta = preprocessor.preprocess_youtube_comment(
            text,
            check_spam=False,
            check_lang=False,
            profile="transformer",
        )

        self.assertFalse(classical_meta["filtered"])
        self.assertFalse(transformer_meta["filtered"])
        self.assertEqual(classical_meta["processing_profile"], "classical")
        self.assertEqual(transformer_meta["processing_profile"], "transformer")
        self.assertIn("WOW!!!", transformer)
        self.assertNotIn("!", classical)
        self.assertNotIn("#", classical)
        self.assertIn("uppercase_ratio", transformer_meta["text_features"])
        self.assertGreaterEqual(transformer_meta["text_features"]["emoji_count"], 1)


class TransformerTrainingScriptTests(SimpleTestCase):
    def test_resolve_text_column_prefers_canonical_text(self):
        resolved = resolve_text_column(
            ["label", "text_transformer", "text", "text_classical"]
        )
        self.assertEqual(resolved, "text")

    def test_load_split_metadata_and_provenance_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            train_csv = tmp_path / "train.csv"
            meta_path = tmp_path / "split_metadata.json"
            train_csv.write_text("text,label\nhello,Positive\n", encoding="utf-8")
            meta_path.write_text(
                json.dumps(
                    {
                        "split": {
                            "strategy": "group",
                            "random_state": 42,
                            "rows": {"train": 10, "val": 3, "test": 4},
                        },
                        "youtube_preprocess": {
                            "primary_text_profile": "transformer",
                        },
                    }
                ),
                encoding="utf-8",
            )

            metadata, resolved_path = load_split_metadata(train_csv)
            summary = summarize_split_provenance(metadata)

            self.assertEqual(resolved_path, meta_path)
            self.assertEqual(summary, "group_seed42_train10_val3_test4_transformer")

    def test_encoder_spec_normalizes_aliases(self):
        spec = get_encoder_spec("deberta-v3")
        self.assertEqual(spec.key, "deberta_v3")
        self.assertIn("DeBERTa", spec.description)


class CalibrationUtilsTests(SimpleTestCase):
    def test_apply_temperature_to_logits_returns_valid_probabilities(self):
        logits = np.array([[2.0, 1.0, 0.5], [0.1, 0.2, 0.3]])
        probs = apply_temperature_to_logits(logits, 1.7)
        self.assertEqual(probs.shape, logits.shape)
        self.assertTrue(np.allclose(probs.sum(axis=1), np.ones(logits.shape[0])))

    def test_temperature_artifact_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "temperature_scaling.json"
            payload = {"temperature": 1.42, "method": "temperature_scaling"}
            save_temperature_artifact(artifact_path, payload)
            loaded = load_temperature_artifact(artifact_path)
            self.assertEqual(loaded["temperature"], 1.42)
            self.assertEqual(loaded["method"], "temperature_scaling")


class SettingsResolutionTests(SimpleTestCase):
    def test_resolve_runtime_settings_uses_local_defaults_for_development(self):
        settings_data = resolve_runtime_settings(
            {"DJANGO_ENV": "development"},
        )

        self.assertTrue(settings_data["debug"])
        self.assertEqual(settings_data["secret_key"], DEV_SECRET_KEY)
        self.assertEqual(
            settings_data["allowed_hosts"],
            ["localhost", "127.0.0.1"],
        )
        self.assertEqual(
            settings_data["cors_allowed_origins"],
            ["http://localhost:3000", "http://127.0.0.1:3000"],
        )

    def test_resolve_runtime_settings_requires_secret_key_in_production(self):
        with self.assertRaisesMessage(
            ImproperlyConfigured,
            "SECRET_KEY must be set when DEBUG is False.",
        ):
            resolve_runtime_settings(
                {
                    "DJANGO_ENV": "production",
                    "ALLOWED_HOSTS": "api.example.com",
                    "CORS_ALLOWED_ORIGINS": "https://app.example.com",
                },
            )

    def test_resolve_runtime_settings_requires_allowed_hosts_in_production(self):
        with self.assertRaisesMessage(
            ImproperlyConfigured,
            "ALLOWED_HOSTS must be set when DEBUG is False.",
        ):
            resolve_runtime_settings(
                {
                    "DJANGO_ENV": "production",
                    "SECRET_KEY": "prod-secret",
                    "CORS_ALLOWED_ORIGINS": "https://app.example.com",
                },
            )

    def test_resolve_runtime_settings_requires_cors_origins_in_production(self):
        with self.assertRaisesMessage(
            ImproperlyConfigured,
            "CORS_ALLOWED_ORIGINS must be set when DEBUG is False.",
        ):
            resolve_runtime_settings(
                {
                    "DJANGO_ENV": "production",
                    "SECRET_KEY": "prod-secret",
                    "ALLOWED_HOSTS": "api.example.com",
                },
            )

    def test_resolve_runtime_settings_rejects_allow_all_cors_in_production(self):
        with self.assertRaisesMessage(
            ImproperlyConfigured,
            "CORS_ALLOW_ALL_ORIGINS cannot be enabled when DEBUG is False.",
        ):
            resolve_runtime_settings(
                {
                    "DJANGO_ENV": "production",
                    "SECRET_KEY": "prod-secret",
                    "ALLOWED_HOSTS": "api.example.com",
                    "CORS_ALLOW_ALL_ORIGINS": "true",
                },
            )
