from django.urls import path
from .views import registration_view, logout_view, get_user


urlpatterns = [
    path('register/', registration_view, name="register"),
    # Login is issued only via /api/token/ (app_api.MyTokenObtainPairView) —
    # this used to duplicate that with an identical JWT pair, which just
    # doubled the auth surface without adding capability.
    path('logout/', logout_view, name="logout"),
    path('me/<int:id>', get_user, name="get_user"),
]
