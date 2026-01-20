from django.urls import path
from rest_framework import views
# from .views import CustomUserCreate
from .views import *


urlpatterns = [
    path('register/', registration_view, name="register"),
    # path('login/', obtain_auth_token, name="login"),
    path('login/', login_view, name="login"),
    path('logout/', logout_view, name="logout"),
    path('me/<int:id>', get_user, name="get_user"),
    # path('search/', search_keywords, name="search_keywords"),
]