from django.core.management.base import BaseCommand

from app.models import AnalysisJob


class Command(BaseCommand):
    """
    Mark abandoned `youtube/analyze/` background jobs as failed.

    A job's Django-Q2 worker can die without ever updating its row (process
    restart/crash/OOM), leaving it stuck at status="running"/"pending"
    forever — see settings.STALE_ANALYSIS_JOB_TIMEOUT. `app/views.py`'s poll
    endpoint already self-heals a stale job the next time its owner polls it,
    but a job nobody ever polls again (closed tab, abandoned session) is only
    ever cleaned up by this command. This is a sweep of our own AnalysisJob
    rows, not something Django-Q2 does for you (it only tracks its own
    internal task queue, not this app's job status model), so it still needs
    to be run periodically by an external scheduler (cron, systemd timer,
    Windows Task Scheduler).

    Usage:
        python manage.py cleanup_stale_analysis_jobs
    """

    help = "Mark AnalysisJob rows abandoned by a dead worker thread as failed."

    def handle(self, *args, **options):
        count = AnalysisJob.sweep_stale_jobs()
        if count:
            self.stdout.write(self.style.SUCCESS(f"Marked {count} stale job(s) as failed."))
        else:
            self.stdout.write("No stale jobs found.")
