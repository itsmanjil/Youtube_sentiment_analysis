from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from app.models import YouTubeAnalysis, YouTubeVideo
from users.models import NewUser


class UserProfileAPITests(APITestCase):
    def setUp(self):
        self.user = NewUser.objects.create_user(
            email="test@example.com",
            user_name="testuser",
            first_name="Test",
            last_name="User",
            password="testpassword123",
        )
        self.other_user = NewUser.objects.create_user(
            email="other@example.com",
            user_name="otheruser",
            first_name="Other",
            last_name="User",
            password="testpassword123",
        )
        self.video = YouTubeVideo.objects.create(
            video_id="v1",
            title="Video 1",
            channel_name="Channel 1",
            published_at="2026-01-01T00:00:00Z",
        )
        YouTubeAnalysis.objects.create(
            user=self.user,
            video=self.video,
            sentiment_data={"Positive": 1, "Neutral": 0, "Negative": 0},
            total_comments_analyzed=1,
        )

    def test_get_user_requires_authentication(self):
        url = reverse("get_user", kwargs={"id": self.user.id})

        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_get_user_returns_own_profile(self):
        self.client.force_authenticate(user=self.user)
        url = reverse("get_user", kwargs={"id": self.user.id})

        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["email"], self.user.email)
        self.assertEqual(
            response.data["searched_list"],
            [{"video_id": "v1", "title": "Video 1"}],
        )

    def test_get_user_rejects_other_user_profile(self):
        self.client.force_authenticate(user=self.user)
        url = reverse("get_user", kwargs={"id": self.other_user.id})

        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(
            response.data["message"],
            "You can only get your information",
        )
