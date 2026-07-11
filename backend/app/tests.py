import json
import pickle
import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

import numpy as np
from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase, override_settings
from django.urls import reverse
from django.utils import timezone
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
from .models import AnalysisJob, YouTubeVideo, YouTubeAnalysis, YouTubeComment
from .youtube_preprocessor import YouTubePreprocessor
from src.sentiment import SentimentResult
from research.transformers.model_registry import get_encoder_spec
from research.transformers.train_encoder import (
    load_split_metadata,
    resolve_text_column,
    summarize_split_provenance,
)
from research.transformers.export_prob_cube import (
    parse_model_names,
    prepare_scoring_frame,
    resolve_text_column_for_model,
)
from research.route_a.run_encoder_sweep import (
    _best_classical_model,
    _find_mcnemar_row,
)
from research.transformers.calibrate_encoder import (
    LABELS as CALIBRATION_LABELS,
    resolve_model_label_order,
)
from research.transformers.prob_cube_io import (
    load_probability_cube,
    save_probability_cube,
)
from src.utils.calibration import (
    apply_temperature_to_logits,
    load_temperature_artifact,
    save_temperature_artifact,
)
from src.utils.config import Config
from src.utils.runtime_artifacts import (
    get_runtime_artifact_metadata,
    get_runtime_artifact_version,
    load_runtime_artifact_json,
    resolve_runtime_artifact_path,
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

    def batch_analyze(self, texts):
        # All real engines implement batch_analyze (views.py calls it directly
        # for efficiency instead of looping analyze() per comment); the mock
        # must satisfy the same interface.
        return [self.analyze(text) for text in texts]

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

    def _mock_fetcher_with_comments(self, mock_fetcher, comments=None):
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.extract_video_id.return_value = 'HLUamwXQ218'
        mock_fetcher_instance.fetch_video_metadata.return_value = MOCK_VIDEO_METADATA
        mock_fetcher_instance.fetch_comments.return_value = comments or MOCK_COMMENTS_RAW
        return mock_fetcher_instance

    @patch('app.views.YouTubeFetcher')
    @patch('app.views.get_sentiment_engine')
    def test_analyze_video_success_api_mode(self, mock_get_engine, mock_fetcher):
        # Setup mocks
        self._mock_fetcher_with_comments(mock_fetcher)

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
        self.assertEqual(
            analysis.analysis_meta["runtime_artifacts"]["version"],
            get_runtime_artifact_version(),
        )
        # 'filtered.total' must mean "comments actually filtered out" (1, the
        # spam comment) — not len(comments) fetched (4). The frontend
        # displays this field labeled "Total Filtered"; the two numbers were
        # previously conflated across different analyze-related endpoints.
        self.assertEqual(response.data['filtered']['total'], 1)
        self.assertEqual(response.data['filtered']['spam'], 1)
        self.assertEqual(response.data['filtered']['language'], 0)
        self.assertEqual(response.data['filtered']['short'], 0)

    @patch('app.views.YouTubeFetcher')
    @patch('app.views.get_sentiment_engine')
    def test_analyze_video_exposes_uncertainty_stats_in_response_and_analysis_meta(
        self,
        mock_get_engine,
        mock_fetcher,
    ):
        self._mock_fetcher_with_comments(mock_fetcher, comments=MOCK_COMMENTS_RAW[:3])
        mock_get_engine.return_value = MockSentimentEngine()

        response = self.client.post(
            self.analyze_url,
            {
                "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
                "filter_spam": False,
                "filter_language": False,
            },
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        expected = {
            "mean_entropy": 0.0,
            "max_entropy": 0.0,
            "min_entropy": 0.0,
            "high_uncertainty_ratio": 0.0,
        }
        self.assertEqual(response.data["uncertainty_stats"], expected)
        self.assertEqual(response.data["analysis_meta"]["uncertainty_stats"], expected)

    @patch('app.views.YouTubeFetcher')
    @patch('app.views.get_sentiment_engine')
    def test_analyze_video_exposes_calibration_metadata_for_live_engine(
        self,
        mock_get_engine,
        mock_fetcher,
    ):
        self._mock_fetcher_with_comments(mock_fetcher, comments=MOCK_COMMENTS_RAW[:3])
        mock_engine = MockSentimentEngine()
        mock_engine.temperature = 0.9348
        mock_engine.calibration_applied = True
        mock_get_engine.return_value = mock_engine

        response = self.client.post(
            self.analyze_url,
            {
                "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
                "sentiment_model": "logreg",
                "filter_spam": False,
                "filter_language": False,
            },
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data["analysis_meta"]["calibration"],
            {"temperature": 0.9348, "applied": True},
        )

    @patch('app.views.YouTubeFetcher')
    @patch('app.views.get_sentiment_engine')
    def test_analyze_video_persists_ensemble_weights_source(
        self,
        mock_get_engine,
        mock_fetcher,
    ):
        self._mock_fetcher_with_comments(mock_fetcher, comments=MOCK_COMMENTS_RAW[:3])
        mock_engine = MockSentimentEngine()
        mock_engine.requested_models = ["logreg", "svm", "tfidf"]
        mock_engine.base_models = ["logreg", "svm", "tfidf"]
        mock_engine.weights = {"logreg": 0.916, "svm": 0.003, "tfidf": 0.081}
        mock_engine.weights_source = "nsga2"
        mock_engine.model_errors = {}
        mock_engine.temperature = 0.9348
        mock_engine.calibration_applied = True
        mock_get_engine.return_value = mock_engine

        response = self.client.post(
            self.analyze_url,
            {
                "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
                "sentiment_model": "ensemble",
                "ensemble_weights_optimization": "nsga2",
                "filter_spam": False,
                "filter_language": False,
            },
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data["analysis_meta"]["ensemble"]["weights_source"],
            "nsga2",
        )
        self.assertEqual(
            response.data["analysis_meta"]["ensemble"]["weights_optimization_requested"],
            "nsga2",
        )
        mock_get_engine.assert_called_with(
            "ensemble",
            base_models=["logreg", "svm", "tfidf"],
            weights=None,
            weights_optimization="nsga2",
        )

    @patch('app.views.YouTubeFetcher')
    @patch('app.views.get_sentiment_engine')
    def test_analyze_video_exposes_neuro_fuzzy_gate_activation(
        self,
        mock_get_engine,
        mock_fetcher,
    ):
        self._mock_fetcher_with_comments(mock_fetcher, comments=MOCK_COMMENTS_RAW[:3])
        mock_engine = MockSentimentEngine()
        mock_engine.requested_models = ["logreg", "svm", "tfidf"]
        mock_engine.base_models = ["logreg", "svm", "tfidf"]
        mock_engine.mf_type = "gaussian"
        mock_engine.defuzz_method = "centroid"
        mock_engine.t_norm = "min"
        mock_engine.t_conorm = "max"
        mock_engine.alpha_cut = 0.0
        mock_engine.resolution = 100
        mock_engine.confidence_threshold = 0.6
        mock_engine.model_errors = {}
        mock_engine._nf_mfs = {"logreg": [{"center": 0.8, "width": 0.1, "alpha": 1.0}]}
        mock_get_engine.return_value = mock_engine

        response = self.client.post(
            self.analyze_url,
            {
                "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
                "sentiment_model": "fuzzy_ensemble",
                "filter_spam": False,
                "filter_language": False,
            },
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["analysis_meta"]["fuzzy"]["nf_gate_active"])

    @patch('app.views.YouTubeFetcher')
    @patch('app.views.get_sentiment_engine')
    def test_analyze_video_uses_transformer_preprocessing_for_encoder_models(self, mock_get_engine, mock_fetcher):
        self._mock_fetcher_with_comments(mock_fetcher, comments=MOCK_COMMENTS_RAW[:3])

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

    def test_analyze_video_rejects_malformed_url_without_creating_a_job(self):
        jobs_before = AnalysisJob.objects.count()
        data = {"video_url": "not_a_url_at_all"}
        response = self.client.post(self.analyze_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Invalid YouTube URL", response.data["msg"])
        self.assertEqual(AnalysisJob.objects.count(), jobs_before)

    def test_analyze_video_rejects_unknown_sentiment_model_without_creating_a_job(self):
        jobs_before = AnalysisJob.objects.count()
        data = {
            "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
            "sentiment_model": "not_a_real_model",
        }
        response = self.client.post(self.analyze_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Invalid sentiment_model", response.data["msg"])
        self.assertEqual(AnalysisJob.objects.count(), jobs_before)

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

    def test_analyze_video_rejects_out_of_range_fuzzy_resolution(self):
        # `fuzzy_resolution` feeds np.linspace(0, 1, resolution) inside the fuzzy
        # engine; an unbounded value here is a memory-exhaustion vector, so it
        # must be rejected at the request-validation stage like every other
        # bounded numeric parameter.
        data = {
            "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
            "sentiment_model": "fuzzy_ensemble",
            "fuzzy_resolution": 2_000_000_000,
        }

        response = self.client.post(self.analyze_url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("fuzzy_resolution", response.data["msg"])

    def test_analyze_video_rejects_out_of_range_fuzzy_alpha_cut(self):
        data = {
            "video_url": "https://www.youtube.com/watch?v=HLUamwXQ218",
            "sentiment_model": "fuzzy_ensemble",
            "fuzzy_alpha_cut": 5.0,
        }

        response = self.client.post(self.analyze_url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("fuzzy_alpha_cut", response.data["msg"])

    @patch('app.views.YouTubeFetcher')
    def test_analyze_video_api_error_quota(self, mock_fetcher):
        mock_error_content = b'{"error": {"errors": [{"reason": "quotaExceeded"}], "message": "Quota Exceeded"}}'
        mock_resp = MagicMock(status=403)
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.fetch_comments.side_effect = HttpError(resp=mock_resp, content=mock_error_content)

        data = {"video_url": "https://www.youtube.com/watch?v=HLUamwXQ218"}
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
            filtered_spam_count=2,
            filtered_language_count=1,
            filtered_short_count=3,
        )

        url = reverse('app:get_youtube_analysis', kwargs={'video_id': 'v1'})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['video']['id'], 'v1')
        self.assertEqual(response.data['data']['model_used'], 'LOGREG')
        # 'filtered.total' must be present and mean "comments actually
        # filtered out" (2 + 1 + 3 = 6), consistent with the analyze and
        # get_user_youtube_analyses endpoints — this endpoint previously
        # omitted the 'total' key entirely.
        self.assertEqual(response.data['data']['filtered']['total'], 6)

    def test_get_single_analysis_exposes_uncertainty_and_calibration_metadata(self):
        video = YouTubeVideo.objects.create(
            video_id='v2',
            title='Video 2',
            channel_name='Channel 2',
            published_at='2026-01-01T00:00:00Z',
        )
        uncertainty_stats = {
            "mean_entropy": 0.1234,
            "max_entropy": 0.4567,
            "min_entropy": 0.0123,
            "high_uncertainty_ratio": 0.25,
        }
        calibration = {
            "temperature": 0.9348,
            "applied": True,
        }
        YouTubeAnalysis.objects.create(
            user=self.user,
            video=video,
            sentiment_data={'Positive': 4, 'Neutral': 3, 'Negative': 3},
            total_comments_analyzed=10,
            analysis_model='ENSEMBLE',
            analysis_meta={
                "uncertainty_stats": uncertainty_stats,
                "calibration": calibration,
            },
        )

        url = reverse('app:get_youtube_analysis', kwargs={'video_id': 'v2'})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['uncertainty_stats'], uncertainty_stats)
        self.assertEqual(response.data['data']['calibration'], calibration)

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

    def test_health_check_endpoint_reports_real_checks(self):
        url = reverse('app:youtube_health_check')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        body = response.json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['checks']['database'], 'ok')
        self.assertEqual(body['checks']['model_artifacts'], 'ok')

    def test_health_check_endpoint_does_not_require_authentication(self):
        self.client.force_authenticate(user=None)
        url = reverse('app:youtube_health_check')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_health_check_endpoint_reports_missing_model_artifacts(self):
        url = reverse('app:youtube_health_check')
        with patch.object(Path, 'exists', return_value=False):
            response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        body = response.json()
        self.assertEqual(body['status'], 'unhealthy')
        self.assertIn('missing', body['checks']['model_artifacts'])


class YouTubeSearchAPITests(APITestCase):
    def setUp(self):
        self.user = NewUser.objects.create_user(
            email='searchuser@example.com',
            user_name='searchuser',
            first_name='Search',
            last_name='User',
            password='testpassword123',
        )
        self.client.force_authenticate(user=self.user)
        self.search_url = reverse('app:youtube_search')

    @patch('app.views.YouTubeFetcher')
    def test_search_success(self, mock_fetcher):
        mock_fetcher.return_value.search_videos.return_value = [
            {
                'video_id': 'abc123',
                'title': 'Some Video',
                'channel': 'Some Channel',
                'published_at': '2026-01-01T00:00:00Z',
                'thumbnail_url': 'https://i.ytimg.com/vi/abc123/mqdefault.jpg',
            }
        ]

        response = self.client.get(self.search_url, {'q': 'test query'})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']), 1)
        self.assertEqual(response.data['data'][0]['video_id'], 'abc123')
        mock_fetcher.return_value.search_videos.assert_called_once_with('test query', max_results=8)

    @patch('app.views.YouTubeFetcher')
    def test_search_respects_max_results_param(self, mock_fetcher):
        mock_fetcher.return_value.search_videos.return_value = []

        response = self.client.get(self.search_url, {'q': 'test', 'max_results': 3})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_fetcher.return_value.search_videos.assert_called_once_with('test', max_results=3)

    def test_search_requires_query(self):
        response = self.client.get(self.search_url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_search_rejects_overlong_query(self):
        response = self.client.get(self.search_url, {'q': 'x' * 101})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_search_rejects_out_of_range_max_results(self):
        response = self.client.get(self.search_url, {'q': 'test', 'max_results': 100})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch('app.views.YouTubeFetcher')
    def test_search_quota_exceeded(self, mock_fetcher):
        mock_error_content = b'{"error": {"errors": [{"reason": "quotaExceeded"}], "message": "Quota Exceeded"}}'
        mock_resp = MagicMock(status=403)
        mock_fetcher.return_value.search_videos.side_effect = HttpError(
            resp=mock_resp, content=mock_error_content
        )

        response = self.client.get(self.search_url, {'q': 'test'})

        self.assertEqual(response.status_code, status.HTTP_429_TOO_MANY_REQUESTS)
        self.assertIn("quota exceeded", response.data['msg'])

    def test_search_requires_authentication(self):
        self.client.force_authenticate(user=None)
        response = self.client.get(self.search_url, {'q': 'test'})
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class AnalysisJobAsyncAPITests(APITestCase):
    """
    ANALYSIS_RUN_SYNC defaults to True in the test environment (see
    core/settings.py) so the bulk of the analyze-endpoint tests above can
    assert on the response body directly, exactly like the pre-job-queue
    behavior. These tests explicitly flip that off to cover the actual
    async/background-job path (app/models.py::AnalysisJob) that runs in
    dev/production.
    """

    def setUp(self):
        self.user = NewUser.objects.create_user(
            email='async@example.com',
            user_name='asyncuser',
            first_name='Async',
            last_name='User',
            password='testpassword123',
        )
        self.client.force_authenticate(user=self.user)
        self.analyze_url = reverse('app:youtube_analyze')

    def _mock_fetcher_with_comments(self, mock_fetcher, comments=None):
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.extract_video_id.return_value = 'HLUamwXQ218'
        mock_fetcher_instance.fetch_video_metadata.return_value = MOCK_VIDEO_METADATA
        mock_fetcher_instance.fetch_comments.return_value = comments or MOCK_COMMENTS_RAW
        return mock_fetcher_instance

    @staticmethod
    def _run_thread_target_immediately(target=None, args=(), daemon=None):
        """
        Stand-in for threading.Thread that runs the target synchronously in
        the calling (test) thread instead of a real background thread, so
        assertions don't need to poll/sleep for a race-prone real thread to
        finish — the job is fully done by the time `.start()` returns.
        """

        class _ImmediateThread:
            def start(self):
                target(*args)

        return _ImmediateThread()

    @override_settings(ANALYSIS_RUN_SYNC=False)
    @patch('app.views.threading.Thread')
    @patch('app.views.YouTubeFetcher')
    @patch('app.views.get_sentiment_engine')
    def test_analyze_video_returns_202_with_job_id_when_async(
        self, mock_get_engine, mock_fetcher, mock_thread_cls
    ):
        mock_thread_cls.side_effect = self._run_thread_target_immediately
        self._mock_fetcher_with_comments(mock_fetcher)
        mock_get_engine.return_value = MockSentimentEngine()

        response = self.client.post(
            self.analyze_url,
            {"video_url": "https://www.youtube.com/watch?v=HLUamwXQ218"},
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertIn("job_id", response.data)
        self.assertEqual(response.data["status"], AnalysisJob.STATUS_PENDING)

        job = AnalysisJob.objects.get(id=response.data["job_id"])
        self.assertEqual(job.status, AnalysisJob.STATUS_DONE)
        self.assertEqual(job.user, self.user)
        self.assertIn("sentiment_data", job.result)

    @override_settings(ANALYSIS_RUN_SYNC=False)
    @patch('app.views.threading.Thread')
    def test_analyze_video_job_failure_recorded_on_job_not_response(self, mock_thread_cls):
        mock_thread_cls.side_effect = self._run_thread_target_immediately

        response = self.client.post(self.analyze_url, {}, format='json')

        # Parameter validation still happens synchronously before a job is
        # even created — a missing video_url is a 400 with no job_id, same
        # as the fully-synchronous path.
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertNotIn("job_id", response.data)

    @override_settings(ANALYSIS_RUN_SYNC=False)
    @patch('app.views.threading.Thread')
    @patch('app.views.YouTubeFetcher')
    def test_analyze_video_polling_status_endpoint_reflects_job_result(
        self, mock_fetcher, mock_thread_cls
    ):
        mock_thread_cls.side_effect = self._run_thread_target_immediately
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.extract_video_id.return_value = 'HLUamwXQ218'
        mock_fetcher_instance.fetch_video_metadata.return_value = None  # video not found

        response = self.client.post(
            self.analyze_url,
            {"video_url": "https://www.youtube.com/watch?v=HLUamwXQ218"},
            format='json',
        )
        job_id = response.data["job_id"]

        status_url = reverse('app:analysis_job_status', kwargs={'job_id': job_id})
        status_response = self.client.get(status_url)

        self.assertEqual(status_response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(status_response.data['status'], AnalysisJob.STATUS_FAILED)
        self.assertIn('msg', status_response.data)

    def test_job_status_scoped_to_owning_user_not_found_for_others(self):
        other_user = NewUser.objects.create_user(
            email='otherasync@example.com',
            user_name='otherasyncuser',
            first_name='Other',
            last_name='User',
            password='testpassword123',
        )
        job = AnalysisJob.objects.create(user=other_user, request_params={})

        status_url = reverse('app:analysis_job_status', kwargs={'job_id': job.id})
        response = self.client.get(status_url)

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_job_status_unknown_id_returns_404(self):
        status_url = reverse('app:analysis_job_status', kwargs={'job_id': 999999})
        response = self.client.get(status_url)

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def _make_abandoned_job(self, *, status_value=AnalysisJob.STATUS_RUNNING):
        """A job whose worker thread died without ever updating it again —
        `updated_at` is older than settings.STALE_ANALYSIS_JOB_TIMEOUT."""
        from django.conf import settings as django_settings

        job = AnalysisJob.objects.create(
            user=self.user, request_params={}, status=status_value
        )
        stale_time = timezone.now() - django_settings.STALE_ANALYSIS_JOB_TIMEOUT - timedelta(minutes=1)
        AnalysisJob.objects.filter(id=job.id).update(updated_at=stale_time)
        job.refresh_from_db()
        return job

    def test_polling_a_stale_running_job_self_heals_to_failed(self):
        job = self._make_abandoned_job()

        status_url = reverse('app:analysis_job_status', kwargs={'job_id': job.id})
        response = self.client.get(status_url)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.data['status'], AnalysisJob.STATUS_FAILED)
        job.refresh_from_db()
        self.assertEqual(job.status, AnalysisJob.STATUS_FAILED)

    def test_polling_a_fresh_running_job_is_left_alone(self):
        job = AnalysisJob.objects.create(
            user=self.user, request_params={}, status=AnalysisJob.STATUS_RUNNING
        )

        status_url = reverse('app:analysis_job_status', kwargs={'job_id': job.id})
        response = self.client.get(status_url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['status'], AnalysisJob.STATUS_RUNNING)
        job.refresh_from_db()
        self.assertEqual(job.status, AnalysisJob.STATUS_RUNNING)

    def test_sweep_stale_jobs_marks_abandoned_jobs_failed_and_leaves_others(self):
        stale_running = self._make_abandoned_job(status_value=AnalysisJob.STATUS_RUNNING)
        stale_pending = self._make_abandoned_job(status_value=AnalysisJob.STATUS_PENDING)
        fresh_running = AnalysisJob.objects.create(
            user=self.user, request_params={}, status=AnalysisJob.STATUS_RUNNING
        )
        already_done = AnalysisJob.objects.create(
            user=self.user,
            request_params={},
            status=AnalysisJob.STATUS_DONE,
            result={"msg": "ok"},
        )

        count = AnalysisJob.sweep_stale_jobs()

        self.assertEqual(count, 2)
        stale_running.refresh_from_db()
        stale_pending.refresh_from_db()
        fresh_running.refresh_from_db()
        already_done.refresh_from_db()
        self.assertEqual(stale_running.status, AnalysisJob.STATUS_FAILED)
        self.assertEqual(stale_pending.status, AnalysisJob.STATUS_FAILED)
        self.assertEqual(fresh_running.status, AnalysisJob.STATUS_RUNNING)
        self.assertEqual(already_done.status, AnalysisJob.STATUS_DONE)


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


class LiveWiringEngineTests(SimpleTestCase):
    def test_ensemble_runtime_switches_between_pso_and_nsga2_weights(self):
        from src.sentiment.engines.ensemble_engine import EnsembleSentimentEngine

        def _artifact_loader(name):
            artifacts = {
                "temperature_scaling": {"models": []},
                "pso_ensemble_weights": {
                    "weights": {"logreg": 0.2, "svm": 0.3, "tfidf": 0.5}
                },
                "multi_objective_ensemble": {
                    "knee_point": {
                        "weights": {"logreg": 0.7, "svm": 0.2, "tfidf": 0.1}
                    }
                },
            }
            return artifacts.get(name, {})

        with patch(
            "src.sentiment.factory.get_base_engine",
            return_value=MagicMock(),
        ), patch(
            "src.sentiment.engines.ensemble_engine.load_runtime_artifact_json",
            side_effect=_artifact_loader,
        ):
            pso_engine = EnsembleSentimentEngine(weights_optimization="pso")
            nsga2_engine = EnsembleSentimentEngine(weights_optimization="nsga2")

        self.assertEqual(pso_engine.weights_source, "pso")
        self.assertEqual(nsga2_engine.weights_source, "nsga2")
        self.assertAlmostEqual(pso_engine.weights["tfidf"], 0.5)
        self.assertAlmostEqual(nsga2_engine.weights["logreg"], 0.7)

    def test_ensemble_temperature_is_scoped_per_weight_variant(self):
        # results/temperature_scaling.json now fits an independent temperature
        # per served ensemble variant ("ensemble_pso" / "ensemble_nsga2") since
        # research/ci/temperature_scaling.py::score_model scores each variant
        # separately. Applying one variant's temperature to a differently
        # weighted blend rescales probabilities the temperature was never fit
        # on, while still reporting calibration_applied=True — silently wrong
        # calibration metadata. Only the matching variant's row may be used;
        # a variant with no matching row (nsga2 here) or request-supplied
        # weights fall back to uncalibrated (T=1.0).
        from src.sentiment.engines.ensemble_engine import EnsembleSentimentEngine

        def _artifact_loader(name):
            artifacts = {
                "temperature_scaling": {
                    "models": [{"model": "ensemble_pso", "temperature": 0.9348}]
                },
                "pso_ensemble_weights": {
                    "weights": {"logreg": 0.2, "svm": 0.3, "tfidf": 0.5}
                },
                "multi_objective_ensemble": {
                    "knee_point": {
                        "weights": {"logreg": 0.7, "svm": 0.2, "tfidf": 0.1}
                    }
                },
            }
            return artifacts.get(name, {})

        with patch(
            "src.sentiment.factory.get_base_engine",
            return_value=MagicMock(),
        ), patch(
            "src.sentiment.engines.ensemble_engine.load_runtime_artifact_json",
            side_effect=_artifact_loader,
        ):
            pso_engine = EnsembleSentimentEngine(weights_optimization="pso")
            nsga2_engine = EnsembleSentimentEngine(weights_optimization="nsga2")
            custom_engine = EnsembleSentimentEngine(
                weights={"logreg": 0.5, "svm": 0.5, "tfidf": 0.0}
            )

        self.assertTrue(pso_engine.calibration_applied)
        self.assertAlmostEqual(pso_engine.temperature, 0.9348)

        # No "ensemble_nsga2" row in the mocked artifact -> uncalibrated.
        self.assertFalse(nsga2_engine.calibration_applied)
        self.assertEqual(nsga2_engine.temperature, 1.0)

        self.assertFalse(custom_engine.calibration_applied)
        self.assertEqual(custom_engine.temperature, 1.0)

    def test_hybrid_dl_runtime_remains_uncalibrated_without_artifact_row(self):
        from src.sentiment.engines.hybrid_dl_engine import HybridDLSentimentEngine

        fake_torch = ModuleType("torch")
        fake_torch.__path__ = []
        fake_torch.long = "long"
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False)
        )
        fake_torch.device = lambda name: name
        fake_torch.load = lambda path, map_location=None, weights_only=False: {
            "model_state_dict": {"weights": [1.0]}
        }

        fake_torch_nn = ModuleType("torch.nn")
        fake_torch_nn.__path__ = []
        fake_torch_nn_functional = ModuleType("torch.nn.functional")

        fake_hybrid_module = ModuleType(
            "research.architectures.hybrid_cnn_bilstm"
        )

        class FakeHybridCNNBiLSTM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def load_state_dict(self, state_dict):
                self.state_dict = state_dict

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.was_evaluated = True

        fake_hybrid_module.HybridCNNBiLSTM = FakeHybridCNNBiLSTM

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_path = temp_path / "hybrid.pt"
            vocab_path = temp_path / "vocab.pkl"
            model_path.write_bytes(b"stub")
            with open(vocab_path, "wb") as handle:
                pickle.dump(
                    {"word2idx": {"<PAD>": 0, "<UNK>": 1, "hello": 2}},
                    handle,
                )

            with patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "torch.nn": fake_torch_nn,
                    "torch.nn.functional": fake_torch_nn_functional,
                    "research.architectures.hybrid_cnn_bilstm": fake_hybrid_module,
                },
            ), patch(
                "src.sentiment.engines.hybrid_dl_engine.load_runtime_artifact_json",
                return_value={"models": [{"model": "logreg", "temperature": 0.9}]},
            ):
                engine = HybridDLSentimentEngine(
                    model_path=model_path,
                    vocab_path=vocab_path,
                    device="cpu",
                )

        self.assertEqual(engine.temperature, 1.0)
        self.assertFalse(engine.calibration_applied)

    def test_fuzzy_runtime_only_activates_nf_gate_for_matching_model_set(self):
        from src.sentiment.engines.fuzzy_engine import FuzzyEnsembleSentimentEngine

        fake_fuzzy_module = ModuleType(
            "research.computational_intelligence.fuzzy.engine_integration"
        )

        class FakeFuzzySentimentEngine:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_fuzzy_module.FuzzySentimentEngine = FakeFuzzySentimentEngine

        nf_artifact = {
            "architecture": {"model_names": ["logreg", "svm", "tfidf"]},
            "learned_mfs": [
                {
                    "model": "logreg",
                    "center": 0.8,
                    "width": 0.1,
                    "alpha": 1.0,
                }
            ],
        }

        with patch.dict(
            sys.modules,
            {
                "research.computational_intelligence.fuzzy.engine_integration": (
                    fake_fuzzy_module
                )
            },
        ), patch(
            "src.sentiment.factory.get_base_engine",
            return_value=MagicMock(),
        ), patch(
            "src.sentiment.engines.fuzzy_engine.load_runtime_artifact_json",
            return_value=nf_artifact,
        ):
            matching_engine = FuzzyEnsembleSentimentEngine(
                base_models=["logreg", "svm", "tfidf"]
            )
            partial_engine = FuzzyEnsembleSentimentEngine(
                base_models=["logreg", "svm"]
            )

        self.assertTrue(bool(matching_engine._nf_mfs))
        self.assertFalse(bool(partial_engine._nf_mfs))


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


class YouTubeScraperLikesParsingTests(SimpleTestCase):
    def _parser(self):
        from app.youtube_scraper import YouTubeScraper
        return YouTubeScraper()

    def test_parses_plain_and_abbreviated_vote_strings(self):
        # `youtube-comment-downloader` returns `votes` as a string (plain
        # digits or "1.2K"/"3M"-style abbreviations for popular comments),
        # not an int. Passing that straight through to `item['likes']`
        # crashes the like-weighting comparison (`likes > 0`) and the
        # `YouTubeComment.likes` IntegerField save — after the expensive
        # fetch/preprocess/inference steps have already run.
        parser = self._parser()
        self.assertEqual(parser._parse_likes("42"), 42)
        self.assertEqual(parser._parse_likes("1.2K"), 1200)
        self.assertEqual(parser._parse_likes("3M"), 3_000_000)
        self.assertEqual(parser._parse_likes(""), 0)
        self.assertEqual(parser._parse_likes(None), 0)
        self.assertEqual(parser._parse_likes("not a number"), 0)
        self.assertEqual(parser._parse_likes(7), 7)


class YouTubeFetcherErrorHandlingTests(SimpleTestCase):
    """
    Regression guard for a fixed API-key leak: `googleapiclient`'s
    HttpError.__str__/__repr__ always embeds the full request URI, which for
    this client includes `key=<YOUTUBE_API_KEY>` as a query parameter (the
    discovery client is built with `developerKey=`). `YouTubeFetcher` must
    let `HttpError` propagate unmodified rather than re-wrapping it into
    `RuntimeError(f"...: {str(e)}")`, which would bake the leaking URI into a
    new exception message and also bypass the caller's (app/views.py)
    structured, sanitized error classification.

    These tests construct a real `YouTubeFetcher` with `self.youtube` swapped
    for a mock resource that raises `HttpError` from `.execute()` — unlike
    `YouTubeAnalysisAPITests`, which mocks the entire `YouTubeFetcher` class
    and therefore never exercises these methods' real bodies.
    """

    def _fetcher_with_mock_resource(self):
        from app.youtube_fetcher import YouTubeFetcher

        with patch.dict("os.environ", {"YOUTUBE_API_KEY": "test-key"}):
            with patch("app.youtube_fetcher.build") as mock_build:
                fetcher = YouTubeFetcher()
        mock_youtube = MagicMock()
        fetcher.youtube = mock_youtube
        return fetcher, mock_youtube

    def _http_error_with_leaking_uri(self):
        resp = MagicMock(status=403)
        content = b'{"error": {"errors": [{"reason": "quotaExceeded"}], "message": "Quota Exceeded"}}'
        return HttpError(
            resp=resp,
            content=content,
            uri="https://www.googleapis.com/youtube/v3/videos?id=abc&key=SECRET_API_KEY",
        )

    def test_fetch_video_metadata_propagates_raw_http_error(self):
        fetcher, mock_youtube = self._fetcher_with_mock_resource()
        mock_youtube.videos.return_value.list.return_value.execute.side_effect = (
            self._http_error_with_leaking_uri()
        )

        with self.assertRaises(HttpError):
            fetcher.fetch_video_metadata("abc12345678")

    def test_fetch_comments_propagates_raw_http_error(self):
        fetcher, mock_youtube = self._fetcher_with_mock_resource()
        mock_youtube.commentThreads.return_value.list.return_value.execute.side_effect = (
            self._http_error_with_leaking_uri()
        )

        with self.assertRaises(HttpError):
            fetcher.fetch_comments("https://www.youtube.com/watch?v=abc12345678")

    def test_video_metadata_error_never_becomes_a_key_leaking_runtime_error(self):
        """
        The specific regression: before the fix, this failure surfaced as
        `RuntimeError(f"Failed to fetch video metadata: {str(http_error)}")`,
        and `str(http_error)` contains the API key. Confirm it's no longer
        wrapped into any exception whose message embeds the URI/key.
        """
        fetcher, mock_youtube = self._fetcher_with_mock_resource()
        mock_youtube.videos.return_value.list.return_value.execute.side_effect = (
            self._http_error_with_leaking_uri()
        )

        try:
            fetcher.fetch_video_metadata("abc12345678")
            self.fail("Expected an exception")
        except RuntimeError as exc:
            self.fail(
                f"fetch_video_metadata wrapped HttpError into a RuntimeError "
                f"that may leak the API key: {exc}"
            )
        except HttpError:
            pass  # Expected: the raw HttpError propagates, unwrapped.


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

    def test_resolve_model_label_order_uses_checkpoint_config(self):
        engine = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(
                    id2label={0: "Negative", 1: "Neutral", 2: "Positive"}
                )
            )
        )

        self.assertEqual(resolve_model_label_order(engine), CALIBRATION_LABELS)

    def test_resolve_model_label_order_falls_back_on_invalid_config(self):
        engine = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(
                    id2label={0: "Positive", 1: "Neutral", 2: "Spam"}
                )
            )
        )

        self.assertEqual(resolve_model_label_order(engine), CALIBRATION_LABELS)


class ProbabilityCubeIOTests(SimpleTestCase):
    def test_parse_model_names_normalizes_encoder_aliases(self):
        self.assertEqual(
            parse_model_names("modernbert, deberta-v3, logreg"),
            ["modernbert", "deberta_v3", "logreg"],
        )

    def test_resolve_text_column_for_model_uses_family_specific_defaults(self):
        columns = ["text", "text_classical", "text_transformer", "label"]
        self.assertEqual(
            resolve_text_column_for_model(columns, "logreg"),
            "text_classical",
        )
        self.assertEqual(
            resolve_text_column_for_model(columns, "deberta_v3"),
            "text_transformer",
        )

    def test_prepare_scoring_frame_tracks_model_specific_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "split.csv"
            csv_path.write_text(
                "text,label,text_classical,text_transformer\n"
                "Hello!!!,Positive,hello,Hello!!!\n"
                "Bad :-(,Negative,bad,Bad :-(\n",
                encoding="utf-8",
            )

            df, canonical, model_columns = prepare_scoring_frame(
                csv_path,
                model_names=["deberta_v3", "logreg"],
            )

            self.assertEqual(canonical, "text")
            self.assertEqual(model_columns["deberta_v3"], "text_transformer")
            self.assertEqual(model_columns["logreg"], "text_classical")
            self.assertEqual(len(df), 2)


class RouteASweepHelpersTests(SimpleTestCase):
    def test_best_classical_model_prefers_highest_macro_f1(self):
        metrics = {
            "logreg": {"macro_f1": 0.71},
            "svm": {"macro_f1": 0.73},
            "deberta_v3": {"macro_f1": 0.69},
        }

        self.assertEqual(_best_classical_model(metrics, ["logreg", "svm"]), "svm")

    def test_find_mcnemar_row_handles_reverse_pair(self):
        rows = [
            {"model_a": "svm", "model_b": "neuro_fuzzy", "n01": 7, "n10": 4, "significant": False},
        ]

        row = _find_mcnemar_row(rows, "neuro_fuzzy", "svm")

        self.assertEqual(row["model_a"], "neuro_fuzzy")
        self.assertEqual(row["model_b"], "svm")
        self.assertEqual(row["n01"], 4)
        self.assertEqual(row["n10"], 7)

    def test_probability_cube_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "cube.npz"
            prob_cube = np.array(
                [
                    [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]],
                    [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
                ],
                dtype=np.float32,
            )
            logits_cube = np.array(
                [
                    [[3.0, 1.0, 0.2], [0.1, 1.0, 2.4]],
                    [[2.5, 1.4, 0.1], [0.4, 1.1, 1.8]],
                ],
                dtype=np.float32,
            )

            save_probability_cube(
                artifact_path,
                prob_cube=prob_cube,
                logits_cube=logits_cube,
                model_names=["modernbert", "deberta_v3"],
                labels=["Positive", "Neutral", "Negative"],
                y_true=["Positive", "Negative"],
                texts=["great video", "terrible upload"],
                sample_ids=["c1", "c2"],
                metadata={"split": "test", "calibration_profile": "auto"},
            )
            bundle = load_probability_cube(artifact_path)

            self.assertEqual(bundle.model_names, ["modernbert", "deberta_v3"])
            self.assertEqual(bundle.labels, ["Positive", "Neutral", "Negative"])
            self.assertEqual(bundle.y_true, ["Positive", "Negative"])
            self.assertEqual(bundle.texts, ["great video", "terrible upload"])
            self.assertEqual(bundle.sample_ids, ["c1", "c2"])
            self.assertEqual(bundle.metadata["split"], "test")
            self.assertTrue(np.allclose(bundle.prob_cube, prob_cube))
            self.assertTrue(np.allclose(bundle.logits_cube, logits_cube))


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

    def test_force_test_overrides_a_stray_django_env_in_the_environment(self):
        # Regression guard: the checked-in backend/.env and .env.example both
        # default to DJANGO_ENV=development for local `runserver` use. Without
        # `force_test` winning, running `manage.py test` with that `.env` in
        # place silently resolves to the "development" environment instead of
        # "test" — re-enabling real throttling and async job mode mid test
        # run (this broke 10 of 60 tests before the fix).
        settings_data = resolve_runtime_settings(
            {"DJANGO_ENV": "development"},
            force_test=True,
        )
        self.assertEqual(settings_data["environment"], "test")

    def test_force_test_false_preserves_existing_behavior(self):
        settings_data = resolve_runtime_settings(
            {"DJANGO_ENV": "development"},
            force_test=False,
        )
        self.assertEqual(settings_data["environment"], "development")


class RuntimeArtifactResolverTests(SimpleTestCase):
    def test_runtime_artifact_manifest_resolves_pinned_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_root = Path(temp_dir)
            version = "thesis_live_v1"
            artifact_dir = runtime_root / version
            artifact_dir.mkdir(parents=True, exist_ok=True)

            temperature_path = artifact_dir / "temperature_scaling.json"
            temperature_payload = {
                "models": [
                    {"model": "logreg", "temperature": 1.1111},
                ]
            }
            temperature_path.write_text(json.dumps(temperature_payload))

            manifest_path = artifact_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": version,
                        "artifacts": {
                            "temperature_scaling": {
                                "path": "temperature_scaling.json",
                                "sha256": "dummy",
                            }
                        },
                    }
                )
            )

            with patch.object(Config, "RUNTIME_ARTIFACTS_DIR", runtime_root), patch.object(
                Config,
                "DEFAULT_RUNTIME_ARTIFACT_VERSION",
                version,
            ):
                resolved = resolve_runtime_artifact_path("temperature_scaling")
                metadata = get_runtime_artifact_metadata()
                payload = load_runtime_artifact_json("temperature_scaling")

            self.assertEqual(resolved, temperature_path.resolve())
            self.assertEqual(metadata["version"], version)
            self.assertEqual(metadata["artifacts"]["temperature_scaling"]["sha256"], "dummy")
            self.assertEqual(payload["models"][0]["temperature"], 1.1111)

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
