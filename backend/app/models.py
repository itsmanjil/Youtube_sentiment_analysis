
from django.conf import settings
from django.db import models
from django.utils import timezone
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


class AnalysisJob(models.Model):
    """
    Tracks a `youtube/analyze/` request executed in the background.

    The actual analysis (comment fetch + preprocessing + model inference +
    aspect mining) can take longer than any reasonable HTTP request timeout
    for large `max_comments`/transformer models, so the view creates one of
    these, runs the work on a background thread, and returns immediately;
    the frontend polls `GET youtube/analyze/status/<id>/` until `status` is
    `done` or `failed`. Persisting job state here (rather than an in-memory
    dict) means a status poll works correctly even if it lands on a
    different WSGI worker process than the one running the job.
    """

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_DONE, "Done"),
        (STATUS_FAILED, "Failed"),
    ]

    user = models.ForeignKey(NewUser, on_delete=models.CASCADE, related_name="analysis_jobs")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    # The validated request parameters, so the background thread/task has
    # everything it needs without depending on the original HTTP request
    # object (which doesn't survive past the view function returning).
    request_params = models.JSONField()
    # Populated on success: the exact payload the synchronous endpoint used
    # to return directly in the response body.
    result = models.JSONField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    error_status = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Analysis Job"
        verbose_name_plural = "Analysis Jobs"

    def __str__(self):
        return f"AnalysisJob({self.id}, {self.status})"

    def is_stale(self, *, now=None):
        """
        True if this job is still pending/running but hasn't been touched
        (see `_execute_analysis_job`'s `updated_at`-bumping saves) in longer
        than settings.STALE_ANALYSIS_JOB_TIMEOUT — i.e. its worker thread
        almost certainly died without ever recording a result.
        """
        if self.status not in (self.STATUS_PENDING, self.STATUS_RUNNING):
            return False
        now = now or timezone.now()
        return (now - self.updated_at) > settings.STALE_ANALYSIS_JOB_TIMEOUT

    def mark_stale_failed(self):
        """Mark an abandoned job failed in place. Caller must have already
        confirmed `is_stale()`."""
        self.status = self.STATUS_FAILED
        self.error_message = (
            "Analysis did not complete — the background worker appears to "
            "have stopped unexpectedly. Please try again."
        )
        self.error_status = 500
        self.save(update_fields=["status", "error_message", "error_status", "updated_at"])

    @classmethod
    def sweep_stale_jobs(cls):
        """
        Mark every abandoned pending/running job as failed. Returns the
        number of jobs updated. Safe to call repeatedly/concurrently — each
        job is only ever touched by whichever caller's `is_stale()` check
        observes it first, and marking an already-terminal job is a no-op.
        """
        cutoff = timezone.now() - settings.STALE_ANALYSIS_JOB_TIMEOUT
        stale_jobs = cls.objects.filter(
            status__in=(cls.STATUS_PENDING, cls.STATUS_RUNNING),
            updated_at__lt=cutoff,
        )
        count = 0
        for job in stale_jobs:
            job.mark_stale_failed()
            count += 1
        return count


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
