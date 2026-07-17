from rest_framework import status
from rest_framework.response import Response
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from core.auth_cookies import REFRESH_COOKIE_NAME, set_refresh_cookie


# custom classes
class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims
        token['user_name'] = user.user_name
        token['is_registered'] = user.is_registered
        # ...

        return token


class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

    def finalize_response(self, request, response, *args, **kwargs):
        # Move the refresh token out of the JSON body and into an httpOnly
        # cookie before the response is rendered — see core/auth_cookies.py.
        if response.status_code == 200 and isinstance(response.data, dict):
            refresh = response.data.pop("refresh", None)
            if refresh is not None:
                set_refresh_cookie(response, refresh)
        return super().finalize_response(request, response, *args, **kwargs)


class CookieTokenRefreshView(TokenRefreshView):
    """
    Same as TokenRefreshView, except the refresh token is read from the
    httpOnly cookie set by MyTokenObtainPairView rather than the request
    body, and the rotated refresh token (ROTATE_REFRESH_TOKENS=True) is
    written back to that cookie instead of the JSON body.
    """

    def post(self, request, *args, **kwargs):
        refresh_value = request.data.get("refresh") or request.COOKIES.get(
            REFRESH_COOKIE_NAME
        )
        if not refresh_value:
            # Match SimpleJWT's own "no/invalid token" contract (401 via
            # InvalidToken) rather than letting a missing required field fall
            # through to the serializer's plain ValidationError (400) —
            # a missing refresh cookie is an auth failure, not a bad request.
            raise InvalidToken("No refresh token cookie was provided.")

        serializer = self.get_serializer(data={"refresh": refresh_value})
        try:
            serializer.is_valid(raise_exception=True)
        except TokenError as exc:
            raise InvalidToken(exc.args[0]) from exc

        return Response(serializer.validated_data, status=status.HTTP_200_OK)

    def finalize_response(self, request, response, *args, **kwargs):
        if response.status_code == 200 and isinstance(response.data, dict):
            refresh = response.data.pop("refresh", None)
            if refresh is not None:
                set_refresh_cookie(response, refresh)
        return super().finalize_response(request, response, *args, **kwargs)
