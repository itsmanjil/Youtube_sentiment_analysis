from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from app.models import YouTubeAnalysis, YouTubeVideo
from core.auth_cookies import REFRESH_COOKIE_NAME
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

    def test_get_user_bounds_searched_list_and_avoids_n_plus_one(self):
        # Give the user more analyses than the cap, each on its own video so
        # a naive loop would issue one extra query per row (the N+1 this
        # endpoint used to have) and serialize the entire history.
        for i in range(25):
            video = YouTubeVideo.objects.create(
                video_id=f"vid{i}",
                title=f"Video {i}",
                channel_name="Channel",
                published_at="2026-01-01T00:00:00Z",
            )
            YouTubeAnalysis.objects.create(
                user=self.user,
                video=video,
                sentiment_data={"Positive": 1, "Neutral": 0, "Negative": 0},
                total_comments_analyzed=1,
            )

        self.client.force_authenticate(user=self.user)
        url = reverse("get_user", kwargs={"id": self.user.id})

        # Query count must not scale with history size: user fetch + the
        # single select_related('video') analyses query (+ a small constant
        # for auth/savepoints), NOT one-per-analysis. select_related folds
        # the video join into that one query, so 26 videos add 0 queries.
        with self.assertNumQueries(2):
            response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Capped at 20 most-recent, newest first (setUp added one more "v1").
        self.assertEqual(len(response.data["searched_list"]), 20)
        self.assertEqual(response.data["searched_list"][0]["video_id"], "vid24")


class UserLogoutAPITests(APITestCase):
    # JWT issuance itself (custom claims, etc.) is covered by
    # app_api.tests.JWTAuthAPITests against the single login path,
    # `token_obtain_pair` — login no longer has a second, duplicate route to
    # test here. These tests only exercise logout/blacklisting.
    def setUp(self):
        self.user = NewUser.objects.create_user(
            email="jwtlogin@example.com",
            user_name="jwtloginuser",
            first_name="JWT",
            last_name="Login",
            password="testpassword123",
        )

    def test_logout_requires_refresh_token(self):
        response = self.client.post(reverse("logout"), {}, format="json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data["message"], "refresh token is required")

    def test_logout_blacklists_refresh_token(self):
        # The refresh token lives in an httpOnly cookie now; the test client
        # persists cookies across requests like a browser, so logout needs
        # no body here.
        login_response = self.client.post(
            reverse("token_obtain_pair"),
            {"email": self.user.email, "password": "testpassword123"},
            format="json",
        )
        refresh_token = login_response.cookies[REFRESH_COOKIE_NAME].value

        logout_response = self.client.post(reverse("logout"), {}, format="json")

        self.assertEqual(logout_response.status_code, status.HTTP_200_OK)
        self.assertEqual(logout_response.data["message"], "User logged out")

        # Re-attach the now-blacklisted token explicitly rather than relying
        # on the test client's cookie jar reflecting the server's
        # clear_refresh_cookie() — this confirms it's rejected specifically
        # because it was blacklisted, not merely because the cookie is gone.
        self.client.cookies[REFRESH_COOKIE_NAME] = refresh_token
        refresh_response = self.client.post(
            reverse("token_refresh"), {}, format="json"
        )

        self.assertEqual(refresh_response.status_code, status.HTTP_401_UNAUTHORIZED)
