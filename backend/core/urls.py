from django.contrib import admin
from django.urls import path, include
from app_api.views import CookieTokenRefreshView, MyTokenObtainPairView

urlpatterns = [
    path('admin/', admin.site.urls),

    # JWT Authentication
    # Use custom token views so JWT contains user_name / is_registered claims
    # (frontend decodes these for UI state) and the refresh token is stored
    # in an httpOnly cookie rather than the JSON body (see core/auth_cookies.py).
    path('api/token/', MyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', CookieTokenRefreshView.as_view(), name='token_refresh'),

    # App-specific URLs
    path('api/user/', include('users.urls')),
    path('api/', include('app.urls')),
]
