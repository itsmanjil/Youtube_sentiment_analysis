import datetime
from rest_framework.authentication import TokenAuthentication
from rest_framework.authtoken.models import Token
from rest_framework.exceptions import AuthenticationFailed

from datetime import date, timedelta
from django.utils import timezone
from django.conf import settings
import datetime
import pytz




#DEFAULT_AUTHENTICATION_CLASSES
class ExpiringTokenAuthentication(TokenAuthentication):
    def authenticate_credentials(self, key):
        try:
            token = Token.objects.get(key = key)
        except Token.DoesNotExist:
            raise AuthenticationFailed("Invalid Token!!")
        
        if not token.user.is_active:
            raise AuthenticationFailed("User is not active")

        utc_now = datetime.datetime.utcnow()
        utc_now = utc_now.replace(tzinfo=pytz.utc)

        if token.created < utc_now - datetime.timedelta(seconds=settings.TOKEN_EXPIRED_AFTER_SECONDS):
            raise  AuthenticationFailed("Token has expired...")
        
        return (token.user, token)