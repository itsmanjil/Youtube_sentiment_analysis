from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenRefreshView,
)
from app_api.views import MyTokenObtainPairView

urlpatterns = [
    path('admin/', admin.site.urls),

    # JWT Authentication
    # Use custom token view so JWT contains user_name / is_registered claims
    # (frontend decodes these for UI state).
    path('api/token/', MyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # App-specific URLs
    path('api/user/', include('users.urls')),
    path('api/', include('app.urls')),
]
