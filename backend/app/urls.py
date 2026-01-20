
from django.urls import path
from .views import (
    analyze_youtube_video,
    get_youtube_analysis,
    get_user_youtube_analyses,
    index
)

app_name = 'app'

urlpatterns = [
    path("youtube/analyze/", analyze_youtube_video, name="youtube_analyze"),
    path("youtube/analysis/<str:video_id>/", get_youtube_analysis, name="get_youtube_analysis"),
    path("youtube/analyses/", get_user_youtube_analyses, name="get_user_analyses"),
    path("youtube/health/", index, name="youtube_health_check"),
]
