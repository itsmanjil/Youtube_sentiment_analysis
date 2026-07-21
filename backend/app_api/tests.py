from unittest.mock import patch

from django.core.cache import cache
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from rest_framework.throttling import ScopedRateThrottle
from rest_framework_simplejwt.tokens import AccessToken

from core.auth_cookies import REFRESH_COOKIE_NAME
from users.models import NewUser


class JWTAuthAPITests(APITestCase):
    def setUp(self):
        self.user = NewUser.objects.create_user(
            email="jwt@example.com",
            user_name="jwtuser",
            first_name="JWT",
            last_name="User",
            password="testpassword123",
        )

    def test_token_obtain_pair_is_public_and_includes_custom_claims(self):
        response = self.client.post(
            reverse("token_obtain_pair"),
            {"email": self.user.email, "password": "testpassword123"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("access", response.data)

        access_token = AccessToken(response.data["access"])
        self.assertEqual(access_token["user_name"], self.user.user_name)
        self.assertEqual(access_token["is_registered"], self.user.is_registered)

    def test_token_obtain_pair_sets_httponly_refresh_cookie_not_body(self):
        response = self.client.post(
            reverse("token_obtain_pair"),
            {"email": self.user.email, "password": "testpassword123"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # The refresh token must never appear in the JSON body — that's the
        # whole point of moving it to an httpOnly cookie (a JS-readable body
        # field would defeat the purpose).
        self.assertNotIn("refresh", response.data)

        cookie = response.cookies.get(REFRESH_COOKIE_NAME)
        self.assertIsNotNone(cookie)
        self.assertTrue(cookie["httponly"])
        self.assertEqual(cookie["samesite"], "Lax")
        self.assertEqual(cookie["path"], "/api/")

    def test_token_refresh_reads_cookie_and_rotates_it(self):
        login_response = self.client.post(
            reverse("token_obtain_pair"),
            {"email": self.user.email, "password": "testpassword123"},
            format="json",
        )
        original_cookie = login_response.cookies[REFRESH_COOKIE_NAME].value

        # The test client persists cookies between requests (like a browser),
        # so no body is needed here — the view must read the cookie itself.
        refresh_response = self.client.post(
            reverse("token_refresh"), {}, format="json"
        )

        self.assertEqual(refresh_response.status_code, status.HTTP_200_OK)
        self.assertIn("access", refresh_response.data)
        self.assertNotIn("refresh", refresh_response.data)

        rotated_cookie = refresh_response.cookies.get(REFRESH_COOKIE_NAME)
        self.assertIsNotNone(rotated_cookie)
        self.assertNotEqual(rotated_cookie.value, original_cookie)

    def test_token_refresh_without_cookie_is_rejected(self):
        response = self.client.post(reverse("token_refresh"), {}, format="json")

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_login_endpoint_is_rate_limited(self):
        # Confirms the dedicated 'login' ScopedRateThrottle is actually wired
        # to POST /api/token/. The test env sets the 'login' rate far above
        # 'anon' so the auth suites aren't throttled, so tighten it here.
        # patch.dict (not override_settings) because DRF binds THROTTLE_RATES
        # as a class snapshot at import — a REST_FRAMEWORK override wouldn't
        # reach the already-constructed throttle. The cache backs throttle
        # counters, so clear it around this test to isolate the per-IP count.
        cache.clear()
        self.addCleanup(cache.clear)
        url = reverse("token_obtain_pair")
        creds = {"email": self.user.email, "password": "wrong-password"}

        with patch.dict(ScopedRateThrottle.THROTTLE_RATES, {"login": "2/minute"}):
            first = self.client.post(url, creds, format="json")
            second = self.client.post(url, creds, format="json")
            third = self.client.post(url, creds, format="json")

        # Wrong credentials are 401; the point is that the throttle blocks the
        # third attempt before auth even runs, which is what slows brute force.
        self.assertIn(
            first.status_code,
            (status.HTTP_400_BAD_REQUEST, status.HTTP_401_UNAUTHORIZED),
        )
        self.assertIn(
            second.status_code,
            (status.HTTP_400_BAD_REQUEST, status.HTTP_401_UNAUTHORIZED),
        )
        self.assertEqual(third.status_code, status.HTTP_429_TOO_MANY_REQUESTS)

    def test_refresh_endpoint_is_not_bound_to_the_login_throttle(self):
        # The tight login scope must not apply to token refresh, which legit
        # clients poll roughly once a minute — only the login view carries it.
        from app_api.views import CookieTokenRefreshView, MyTokenObtainPairView

        self.assertEqual(getattr(MyTokenObtainPairView, "throttle_scope", None), "login")
        self.assertNotEqual(
            getattr(CookieTokenRefreshView, "throttle_scope", None), "login"
        )
