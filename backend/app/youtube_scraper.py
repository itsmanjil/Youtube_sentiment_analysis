
import re
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class YouTubeScraper:

    def __init__(self):
        try:
            from youtube_comment_downloader import YoutubeCommentDownloader
            self.downloader = YoutubeCommentDownloader()
        except ImportError:
            raise ImportError(
                "youtube-comment-downloader not installed. "
                "Run: pip install youtube-comment-downloader"
            )

    def extract_video_id(self, url):
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([\w-]{11})',
            r'(?:youtu\.be\/)([\w-]{11})',
            r'(?:youtube\.com\/embed\/)([\w-]{11})',
            r'(?:youtube\.com\/v\/)([\w-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        if len(url) == 11 and re.match(r'^[\w-]{11}$', url):
            return url

        return None

    def _parse_relative_time(self, time_str):
        """
        Parse relative timestamps like '2 days ago', '1 month ago', etc.

        Args:
            time_str (str): Relative time string

        Returns:
            str: ISO format timestamp or None if parsing fails
        """
        if not time_str or time_str.strip() == '':
            return None

        try:
            time_str = time_str.lower().strip()
            if not re.search(r"[a-z]", time_str):
                return None
            now = datetime.now()

            # Parse patterns like "X seconds/minutes/hours/days/weeks/months/years ago"
            patterns = [
                (r'(\d+)\s*second', 'seconds'),
                (r'(\d+)\s*minute', 'minutes'),
                (r'(\d+)\s*hour', 'hours'),
                (r'(\d+)\s*day', 'days'),
                (r'(\d+)\s*week', 'weeks'),
                (r'(\d+)\s*month', 'months'),
                (r'(\d+)\s*year', 'years'),
            ]

            for pattern, unit in patterns:
                match = re.search(pattern, time_str)
                if match:
                    value = int(match.group(1))

                    if unit == 'seconds':
                        delta = timedelta(seconds=value)
                    elif unit == 'minutes':
                        delta = timedelta(minutes=value)
                    elif unit == 'hours':
                        delta = timedelta(hours=value)
                    elif unit == 'days':
                        delta = timedelta(days=value)
                    elif unit == 'weeks':
                        delta = timedelta(weeks=value)
                    elif unit == 'months':
                        delta = timedelta(days=value * 30)  # Approximate
                    elif unit == 'years':
                        delta = timedelta(days=value * 365)  # Approximate
                    else:
                        continue

                    return (now - delta).isoformat()

            # If no pattern matched, return None
            logger.warning(f"Could not parse time string: {time_str}")
            return None

        except Exception as e:
            logger.warning(f"Error parsing time string '{time_str}': {e}")
            return None

    def _scrape_metadata_with_ytdlp(self, video_id):
        """
        Scrape video metadata using yt-dlp.

        Args:
            video_id (str): YouTube video ID

        Returns:
            dict: Video metadata or None if scraping fails
        """
        try:
            import yt_dlp

            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    f'https://www.youtube.com/watch?v={video_id}',
                    download=False
                )

                # Convert upload_date from YYYYMMDD to ISO format
                upload_date = info.get('upload_date', '')
                if upload_date and len(upload_date) == 8:
                    try:
                        date_obj = datetime.strptime(upload_date, '%Y%m%d')
                        upload_date = date_obj.isoformat()
                    except:
                        upload_date = datetime.now().isoformat()
                else:
                    upload_date = datetime.now().isoformat()

                return {
                    'title': info.get('title', f'Video {video_id}'),
                    'description': (info.get('description', '') or '')[:500],  # Limit length
                    'channel': info.get('uploader', 'Unknown'),
                    'channel_id': info.get('channel_id', ''),
                    'published_at': upload_date,
                    'view_count': info.get('view_count', 0) or 0,
                    'like_count': info.get('like_count', 0) or 0,
                    'comment_count': info.get('comment_count', 0) or 0,
                    'thumbnail_url': info.get('thumbnail', f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg')
                }
        except ImportError:
            logger.warning("yt-dlp not installed, cannot scrape metadata")
            return None
        except Exception as e:
            logger.warning(f"yt-dlp failed for video {video_id}: {e}")
            return None

    def _fallback_metadata(self, video_id):
        """
        Graceful fallback metadata when scraping fails.

        Args:
            video_id (str): YouTube video ID

        Returns:
            dict: Minimal fallback metadata
        """
        return {
            'title': f'Video {video_id} (metadata unavailable)',
            'description': '',
            'channel': 'Unknown',
            'channel_id': '',
            'published_at': datetime.now().isoformat(),
            'view_count': 0,
            'like_count': 0,
            'comment_count': 0,
            'thumbnail_url': f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg'
        }

    def fetch_comments(self, video_url, max_results=100, sort_by='top'):
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {video_url}")

        comments = []
        sort_mode = 0 if sort_by == 'top' else 1

        try:
            # Note: youtube-comment-downloader returns a generator
            try:
                comment_iter = self.downloader.get_comments_from_url(
                    f'https://www.youtube.com/watch?v={video_id}',
                    sort_by=sort_mode
                )
            except TypeError:
                comment_iter = self.downloader.get_comments_from_url(
                    f'https://www.youtube.com/watch?v={video_id}',
                    sort=sort_mode
                )

            for comment in comment_iter:
                if len(comments) >= max_results:
                    break

                # Parse the time string if available (fix: use helper function)
                time_str = comment.get('time', '')
                published_at = self._parse_relative_time(time_str)
                # If parsing failed, set to None instead of current time
                if published_at is None:
                    published_at = None  # Will be handled by caller

                # Fix reply detection: check if comment has a parent
                # Top-level comments don't have 'parent' field, replies do
                is_reply = bool(comment.get('parent', ''))

                comment_data = {
                    'text': comment.get('text', ''),
                    'author': comment.get('author', 'Unknown'),
                    'likes': comment.get('votes', 0),
                    'published_at': published_at,
                    'reply_count': 0,  # Not available in scraper
                    'is_reply': is_reply,
                    'video_id': video_id,
                    'comment_id': comment.get('cid', '')
                }
                comments.append(comment_data)

            if not comments:
                raise RuntimeError("No comments found. Video may have comments disabled.")

            return comments

        except Exception as e:
            if "No comments found" in str(e):
                raise RuntimeError(
                    "No comments found. The video may have comments disabled, "
                    "be age-restricted, or the scraper may have been blocked."
                )
            raise RuntimeError(f"Scraper error: {str(e)}")

    def fetch_video_metadata(self, video_id):
        """
        Fetch video metadata using yt-dlp with fallback.

        Args:
            video_id (str): YouTube video ID

        Returns:
            dict: Video metadata
        """
        # Try yt-dlp first
        metadata = self._scrape_metadata_with_ytdlp(video_id)

        if metadata is not None:
            logger.info(f"Successfully scraped metadata for {video_id} using yt-dlp")
            return metadata

        # Fallback to minimal metadata
        logger.warning(f"Using fallback metadata for {video_id}")
        return self._fallback_metadata(video_id)
