
from unittest.mock import patch, MagicMock

from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from googleapiclient.errors import HttpError

from users.models import NewUser
from .analysis_utils import (
    aggregate_confidence_stats,
    bootstrap_confidence_intervals,
    normalize_probs,
)
from .models import YouTubeVideo, YouTubeAnalysis, YouTubeComment
from src.sentiment import SentimentResult

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
        self.assertEqual(analysis.sentiment_data['Negative'], 0)
        self.assertEqual(analysis.sentiment_data['Neutral'], 2)

    def test_analyze_video_missing_url(self):
        data = {"max_comments": 100}
        response = self.client.post(self.analyze_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data['msg'], 'video_url is required')

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
