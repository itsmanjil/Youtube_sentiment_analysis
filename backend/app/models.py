
from django.db import models
from users.models import NewUser


class YouTubeVideo(models.Model):
    """Store YouTube video metadata"""
    video_id = models.CharField(max_length=11, unique=True, db_index=True)
    title = models.TextField()
    description = models.TextField(null=True, blank=True)
    channel_name = models.CharField(max_length=255)
    channel_id = models.CharField(max_length=255, null=True, blank=True)
    published_at = models.DateTimeField()
    view_count = models.BigIntegerField(default=0)
    like_count = models.BigIntegerField(default=0)
    comment_count = models.BigIntegerField(default=0)
    thumbnail_url = models.URLField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} ({self.video_id})"

    class Meta:
        ordering = ['-created_at']
        verbose_name = "YouTube Video"
        verbose_name_plural = "YouTube Videos"


class YouTubeComment(models.Model):
    """Store individual YouTube comments"""
    video = models.ForeignKey(YouTubeVideo, on_delete=models.CASCADE, related_name='comments')
    comment_id = models.CharField(max_length=100, unique=True, null=True, blank=True)
    text = models.TextField()
    author = models.CharField(max_length=255)
    likes = models.IntegerField(default=0)
    published_at = models.DateTimeField()
    is_reply = models.BooleanField(default=False)
    sentiment = models.CharField(max_length=20, null=True, blank=True)
    sentiment_score = models.FloatField(null=True, blank=True)
    is_spam = models.BooleanField(default=False)
    language = models.CharField(max_length=10, default='en')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.author}: {self.text[:50]}..."

    class Meta:
        ordering = ['-likes', '-published_at']
        verbose_name = "YouTube Comment"
        verbose_name_plural = "YouTube Comments"


class YouTubeAnalysis(models.Model):
    """Store YouTube sentiment analysis results"""
    user = models.ForeignKey(NewUser, on_delete=models.SET_NULL, null=True)
    video = models.ForeignKey(YouTubeVideo, on_delete=models.CASCADE, related_name='analyses')
    sentiment_data = models.JSONField()  # {"Positive": 150, "Negative": 50, "Neutral": 100}
    hour_data = models.JSONField(null=True, blank=True)  # Hourly sentiment distribution
    like_weighted_sentiment = models.JSONField(null=True, blank=True)  # Top liked comments by sentiment
    top_words_positive = models.JSONField(null=True, blank=True)  # Word frequency for word clouds
    top_words_negative = models.JSONField(null=True, blank=True)
    analysis_meta = models.JSONField(null=True, blank=True)  # Confidence, aspects, CI metadata
    total_comments_analyzed = models.IntegerField()
    filtered_spam_count = models.IntegerField(default=0)
    filtered_language_count = models.IntegerField(default=0)
    filtered_short_count = models.IntegerField(default=0)
    fetched_date = models.DateTimeField(auto_now_add=True)
    analysis_model = models.CharField(max_length=50, default='LOGREG')  # LOGREG, SVM, TF-IDF, Ensemble

    def __str__(self):
        return f"Analysis of {self.video.title} - {self.fetched_date.strftime('%Y-%m-%d')}"

    class Meta:
        ordering = ['-fetched_date']
        verbose_name = "YouTube Analysis"
        verbose_name_plural = "YouTube Analyses"
