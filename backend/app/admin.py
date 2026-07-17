
from django.contrib import admin
from .models import YouTubeVideo, YouTubeComment, YouTubeAnalysis


@admin.register(YouTubeVideo)
class YouTubeVideoAdmin(admin.ModelAdmin):
    list_display = ('video_id', 'title', 'channel_name', 'view_count', 'like_count', 'created_at')
    search_fields = ('video_id', 'title', 'channel_name')
    list_filter = ('created_at',)
    readonly_fields = ('video_id', 'created_at', 'updated_at')
    ordering = ('-created_at',)


@admin.register(YouTubeComment)
class YouTubeCommentAdmin(admin.ModelAdmin):
    list_display = ('author', 'text_preview', 'sentiment', 'likes', 'video', 'published_at')
    search_fields = ('author', 'text', 'video__title')
    list_filter = ('sentiment', 'is_spam', 'language', 'is_reply')
    readonly_fields = ('comment_id', 'created_at')
    ordering = ('-likes', '-published_at')

    def text_preview(self, obj):
        return obj.text[:50] + "..." if len(obj.text) > 50 else obj.text
    text_preview.short_description = 'Comment'


@admin.register(YouTubeAnalysis)
class YouTubeAnalysisAdmin(admin.ModelAdmin):
    list_display = ('video', 'user', 'total_comments_analyzed', 'analysis_model', 'fetched_date')
    search_fields = ('video__title', 'user__email')
    list_filter = ('analysis_model', 'fetched_date')
    readonly_fields = ('fetched_date',)
    ordering = ('-fetched_date',)
