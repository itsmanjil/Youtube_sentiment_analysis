
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import re
from datetime import datetime


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
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {video_url}")

        comments = []
        next_page_token = None

        try:
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

        except HttpError as e:
            if e.resp.status == 403:
                raise RuntimeError("YouTube API quota exceeded or API key invalid")
            elif e.resp.status == 404:
                raise RuntimeError(f"Video not found: {video_id}")
            elif e.resp.status == 400:
                raise RuntimeError(f"Invalid request: {str(e)}")
            else:
                raise RuntimeError(f"YouTube API error: {str(e)}")

    def fetch_video_metadata(self, video_id):
        try:
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

        except HttpError as e:
            raise RuntimeError(f"Failed to fetch video metadata: {str(e)}")

    def fetch_channel_videos(self, channel_id, max_results=10):
        try:
            request = self.youtube.search().list(
                part='id',
                channelId=channel_id,
                maxResults=max_results,
                order='date',
                type='video'
            )
            response = request.execute()
            self.quota_used += 100  # Search costs 100 units

            video_ids = [item['id']['videoId'] for item in response['items']]
            return video_ids

        except HttpError as e:
            raise RuntimeError(f"Failed to fetch channel videos: {str(e)}")

    def get_quota_used(self):
        return self.quota_used
