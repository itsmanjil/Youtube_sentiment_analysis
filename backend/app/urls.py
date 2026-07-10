
from django.urls import path
from .views import (
    analyze_youtube_video,
    get_analysis_job_status,
    get_youtube_analysis,
    get_user_youtube_analyses,
    search_youtube_videos,
    index
)

app_name = 'app'

urlpatterns = [
    path("youtube/analyze/", analyze_youtube_video, name="youtube_analyze"),
    path("youtube/analyze/status/<int:job_id>/", get_analysis_job_status, name="analysis_job_status"),
    path("youtube/analysis/<str:video_id>/", get_youtube_analysis, name="get_youtube_analysis"),
    path("youtube/analyses/", get_user_youtube_analyses, name="get_user_analyses"),
    path("youtube/search/", search_youtube_videos, name="youtube_search"),
    path("youtube/health/", index, name="youtube_health_check"),
]
