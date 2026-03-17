from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from rest_framework_simplejwt.tokens import AccessToken

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
        self.assertIn("refresh", response.data)

        access_token = AccessToken(response.data["access"])
        self.assertEqual(access_token["user_name"], self.user.user_name)
        self.assertEqual(access_token["is_registered"], self.user.is_registered)
