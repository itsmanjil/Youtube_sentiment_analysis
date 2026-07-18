
from googleapiclient.discovery import build
import os
import re


class YouTubeFetcher:

    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY not found in environment variables")

        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.quota_used = 0

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

        # If no pattern matched, check if it's already a video ID
        if len(url) == 11 and re.match(r'^[\w-]{11}$', url):
            return url

        return None

    def fetch_comments(self, video_url, max_results=100, order='relevance'):
        # Raises HttpError (uncaught) on any Google API failure — the caller
        # (app/views.py::_execute_analysis_job) already has structured error
        # classification (quota/invalid key/comments disabled/not found)
        # keyed off HttpError.resp.status and the parsed JSON error body, and
        # sanitizes what reaches the client. Catching and re-wrapping HttpError
        # here into a plain RuntimeError(str(e)) would both bypass that
        # classification and leak the request URL (HttpError.__str__ embeds
        # `self.uri`, which for this client always includes `key=<API_KEY>`
        # as a query parameter) straight into an exception message.
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {video_url}")

        comments = []
        next_page_token = None

        while len(comments) < max_results:
            # Fetch comment threads (top-level comments)
            request = self.youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_page_token,
                textFormat='plainText',
                order=order
            )
            response = request.execute()
            self.quota_used += 1  # 1 unit per request

            for item in response['items']:
                top_comment = item['snippet']['topLevelComment']['snippet']

                comment_data = {
                    'text': top_comment['textDisplay'],
                    'author': top_comment['authorDisplayName'],
                    'likes': top_comment['likeCount'],
                    'published_at': top_comment['publishedAt'],
                    'reply_count': item['snippet']['totalReplyCount'],
                    'is_reply': False,
                    'video_id': video_id,
                    'comment_id': item['snippet']['topLevelComment']['id']
                }
                comments.append(comment_data)

                # Fetch replies if they exist
                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        if len(comments) >= max_results:
                            break

                        reply_snippet = reply['snippet']
                        reply_data = {
                            'text': reply_snippet['textDisplay'],
                            'author': reply_snippet['authorDisplayName'],
                            'likes': reply_snippet['likeCount'],
                            'published_at': reply_snippet['publishedAt'],
                            'reply_count': 0,
                            'is_reply': True,
                            'video_id': video_id,
                            'comment_id': reply['id']
                        }
                        comments.append(reply_data)

                if len(comments) >= max_results:
                    break

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        return comments[:max_results]

    def fetch_video_metadata(self, video_id):
        # Raises HttpError (uncaught) on failure — see the comment at the top
        # of fetch_comments() for why this must not be wrapped into a
        # RuntimeError(str(e)) here.
        request = self.youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        )
        response = request.execute()
        self.quota_used += 1

        if not response['items']:
            return None

        item = response['items'][0]
        snippet = item['snippet']
        statistics = item['statistics']

        return {
            'title': snippet['title'],
            'description': snippet['description'],
            'channel': snippet['channelTitle'],
            'channel_id': snippet['channelId'],
            'published_at': snippet['publishedAt'],
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'comment_count': int(statistics.get('commentCount', 0)),
            'thumbnail_url': snippet['thumbnails']['high']['url']
        }

    def search_videos(self, query, max_results=10):
        # Raises HttpError (uncaught) on failure — see the comment at the top
        # of fetch_comments() for why this must not be wrapped into a
        # RuntimeError(str(e)) here.
        request = self.youtube.search().list(
            part='snippet',
            q=query,
            maxResults=max_results,
            type='video',
            safeSearch='none',
        )
        response = request.execute()
        self.quota_used += 100  # Search costs 100 units regardless of maxResults

        results = []
        for item in response.get('items', []):
            snippet = item.get('snippet', {})
            video_id = item.get('id', {}).get('videoId')
            if not video_id:
                continue
            thumbnails = snippet.get('thumbnails', {})
            thumbnail = (
                thumbnails.get('medium')
                or thumbnails.get('default')
                or thumbnails.get('high')
                or {}
            )
            results.append({
                'video_id': video_id,
                'title': snippet.get('title', ''),
                'channel': snippet.get('channelTitle', ''),
                'published_at': snippet.get('publishedAt'),
                'thumbnail_url': thumbnail.get('url'),
            })
        return results

    def fetch_channel_videos(self, channel_id, max_results=10):
        # Raises HttpError (uncaught) on failure — see the comment at the top
        # of fetch_comments() for why this must not be wrapped into a
        # RuntimeError(str(e)) here.
        request = self.youtube.search().list(
            part='id',
            channelId=channel_id,
            maxResults=max_results,
            order='date',
            type='video'
        )
        response = request.execute()
        self.quota_used += 100  # Search costs 100 units

        return [item['id']['videoId'] for item in response['items']]

    def get_quota_used(self):
        return self.quota_used
