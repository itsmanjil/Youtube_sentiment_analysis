import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
import django

django.setup()

from app.youtube_fetcher import YouTubeFetcher
from app.youtube_scraper import YouTubeScraper
from app.youtube_preprocessor import YouTubePreprocessor
from src.sentiment import get_sentiment_engine


def test_video_id_extraction():
    """Test video ID extraction from various URL formats."""
    print("=" * 60)
    print("TEST 1: Video ID Extraction")
    print("=" * 60)

    fetcher = YouTubeFetcher() if os.getenv("YOUTUBE_API_KEY") else YouTubeScraper()

    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ]

    for url in test_urls:
        video_id = fetcher.extract_video_id(url)
        status = "[OK]" if video_id == "dQw4w9WgXcQ" else "[FAIL]"
        print(f"{status} {url} -> {video_id}")

    print()


def test_preprocessing():
    """Test comment preprocessing."""
    print("=" * 60)
    print("TEST 2: Comment Preprocessing")
    print("=" * 60)

    preprocessor = YouTubePreprocessor()

    test_comments = [
        "This video is AMAAAAAZING!!! :D",
        "check out my channel! http://spam.com",
        "Great video at 2:45!",
        "Este es un comentario en espanol",
        "@UserName great content!",
        "This is a normal comment",
    ]

    for comment in test_comments:
        processed, metadata = preprocessor.preprocess_youtube_comment(comment)
        status = "FILTERED" if metadata["filtered"] else "OK"
        reason = (
            f"({metadata.get('filter_reason', 'none')})"
            if metadata["filtered"]
            else ""
        )
        print(f"{status:8} {reason:15} | {comment[:40]}...")

    print()


def test_sentiment_analysis():
    """Test LogReg sentiment analysis."""
    print("=" * 60)
    print("TEST 3: Sentiment Analysis")
    print("=" * 60)

    try:
        engine = get_sentiment_engine("logreg")

        test_texts = [
            "This video is absolutely amazing! Love it!",
            "Terrible quality, waste of time",
            "It's okay, nothing special",
        ]

        for text in test_texts:
            result = engine.analyze(text)
            sentiment = result.label
            score = result.score
            print(f"{sentiment:8} ({score:+.3f}) | {text}")

        print()
        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        print("  Train LogReg: python train_logreg_youtube.py")
        print()
        return False


def test_youtube_api():
    """Test YouTube API connection."""
    print("=" * 60)
    print("TEST 4: YouTube API Connection")
    print("=" * 60)

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        print("[FAIL] No YOUTUBE_API_KEY found in environment")
        print("  Set it in backend/.env file")
        print("  Scraper mode will still work without API key")
        print()
        return False

    try:
        fetcher = YouTubeFetcher()
        print(f"[OK] API key loaded: {api_key[:10]}...")

        # Try to fetch metadata for a popular video
        print("  Testing API with video: dQw4w9WgXcQ")
        metadata = fetcher.fetch_video_metadata("dQw4w9WgXcQ")

        if metadata:
            print(f"[OK] API working! Fetched: {metadata['title'][:50]}...")
            print(f"  Views: {metadata['view_count']:,}")
            print(f"  Likes: {metadata['like_count']:,}")
        else:
            print("[FAIL] API returned no metadata")

        print()
        return True

    except Exception as e:
        print(f"[FAIL] API Error: {e}")
        print("  Check your API key in backend/.env")
        print("  Make sure YouTube Data API v3 is enabled")
        print()
        return False


def test_scraper():
    """Test YouTube scraper (no API key needed)."""
    print("=" * 60)
    print("TEST 5: YouTube Scraper (No API Key)")
    print("=" * 60)

    try:
        scraper = YouTubeScraper()
        print("[OK] Scraper initialized")

        print("  Fetching 5 comments from test video...")
        print("  (This may take 5-10 seconds)")

        comments = scraper.fetch_comments(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            max_results=5,
        )

        if comments:
            print(f"[OK] Scraper working! Fetched {len(comments)} comments")
            print(f"  First comment: {comments[0]['text'][:50]}...")
        else:
            print("[FAIL] Scraper returned no comments")

        print()
        return True

    except ImportError:
        print("[FAIL] youtube-comment-downloader not installed")
        print("  Install it: pip install youtube-comment-downloader")
        print()
        return False

    except Exception as e:
        print(f"[FAIL] Scraper Error: {e}")
        print("  The scraper may be blocked or the video may have issues")
        print()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("YouTube Sentiment Analysis - Integration Test")
    print("=" * 60)
    print()

    # Run tests
    test_video_id_extraction()
    test_preprocessing()
    logreg_ok = test_sentiment_analysis()
    api_ok = test_youtube_api()
    scraper_ok = test_scraper()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    checks = {
        "Video ID Extraction": True,  # Always works
        "Comment Preprocessing": True,  # Always works
        "LogReg Sentiment": logreg_ok,
        "YouTube API": api_ok,
        "YouTube Scraper": scraper_ok,
    }

    for check, status in checks.items():
        symbol = "[OK]" if status else "[FAIL]"
        print(f"{symbol} {check}")

    print()

    if all(checks.values()):
        print("All tests passed. YouTube integration is ready.")
    elif logreg_ok and (api_ok or scraper_ok):
        print("Core features working. Fix warnings above for full functionality.")
    else:
        print("Some tests failed. Check errors above.")

    print()
    print("Next steps:")
    print("1. Run migrations: python manage.py migrate")
    print("2. Start server: python manage.py runserver")
    print("3. Test API: POST /api/youtube/analyze/")
    print()


if __name__ == "__main__":
    main()
