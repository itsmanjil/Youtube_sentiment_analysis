from django.urls import path
from .views import registration_view, login_view, logout_view, get_user


urlpatterns = [
    path('register/', registration_view, name="register"),
    path('login/', login_view, name="login"),
    path('logout/', logout_view, name="logout"),
    path('me/<int:id>', get_user, name="get_user"),
]
