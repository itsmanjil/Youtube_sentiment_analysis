from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
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
