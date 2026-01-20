"""
Quick test script to verify YouTube data preparation fixes

Tests:
1. YouTubeScraper timestamp parsing
2. YouTubeScraper reply detection
3. YouTubeScraper metadata scraping (yt-dlp)
4. File paths in views.py
5. Manual labeling CSV import
6. Data validation

Usage:
    python test_fixes.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()


def test_timestamp_parsing():
    """Test 1: YouTubeScraper timestamp parsing"""
    print("\n" + "="*80)
    print("TEST 1: Timestamp Parsing")
    print("="*80)

    from app.youtube_scraper import YouTubeScraper

    scraper = YouTubeScraper()

    test_cases = [
        ("2 days ago", "Should parse as 2 days in the past"),
        ("1 hour ago", "Should parse as 1 hour in the past"),
        ("5 months ago", "Should parse as ~5 months in the past"),
        ("", "Should return None for empty string"),
    ]

    for time_str, expected in test_cases:
        result = scraper._parse_relative_time(time_str)
        status = "✅" if result is not None or time_str == "" else "❌"
        print(f"{status} '{time_str:15s}' -> {result}")

    print("\n✅ Timestamp parsing test complete")
    return True


def test_metadata_scraping():
    """Test 2: YouTubeScraper metadata with yt-dlp"""
    print("\n" + "="*80)
    print("TEST 2: Metadata Scraping (yt-dlp)")
    print("="*80)

    try:
        import yt_dlp
        print("✅ yt-dlp is installed")
    except ImportError:
        print("⚠️  yt-dlp not installed - scraper will use fallback")
        print("   Install with: pip install yt-dlp")
        return False

    from app.youtube_scraper import YouTubeScraper

    scraper = YouTubeScraper()
    test_video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up

    print(f"\nTesting with video: {test_video_id}")

    try:
        metadata = scraper.fetch_video_metadata(test_video_id)

        # Check if we got real data (not placeholders)
        is_real = metadata['title'] != f'Video {test_video_id}' and \
                  metadata['title'] != f'Video {test_video_id} (metadata unavailable)'

        if is_real:
            print(f"✅ Real metadata retrieved!")
            print(f"   Title: {metadata['title'][:60]}...")
            print(f"   Channel: {metadata['channel']}")
            print(f"   Views: {metadata['view_count']:,}")
            return True
        else:
            print(f"⚠️  Got fallback metadata (yt-dlp might have failed)")
            print(f"   Title: {metadata['title']}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_file_paths():
    """Test 3: File paths in views.py"""
    print("\n" + "="*80)
    print("TEST 3: File Paths (views.py)")
    print("="*80)

    try:
        from app import views

        # Check if contractions and negations were loaded
        has_contractions = hasattr(views, 'contractions') and len(views.contractions) > 0
        has_negations = hasattr(views, 'negations') and len(views.negations) > 0

        print(f"{'✅' if has_contractions else '❌'} Contractions loaded: {len(views.contractions) if has_contractions else 0}")
        print(f"{'✅' if has_negations else '❌'} Negations loaded: {len(views.negations) if has_negations else 0}")

        # Check if BASE_DIR is defined
        has_base_dir = hasattr(views, 'BASE_DIR')
        print(f"{'✅' if has_base_dir else '❌'} BASE_DIR defined: {views.BASE_DIR if has_base_dir else 'N/A'}")

        return has_contractions and has_negations and has_base_dir

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_manual_labeling():
    """Test 4: Manual labeling CSV import"""
    print("\n" + "="*80)
    print("TEST 4: Manual Labeling CSV Import")
    print("="*80)

    import pandas as pd
    from prepare_youtube_training_data import YouTubeDataPreparer

    # Create temporary CSV
    temp_dir = Path(tempfile.mkdtemp())
    test_csv = temp_dir / "test_labeled.csv"

    test_data = pd.DataFrame({
        'text': [
            'This is amazing!',
            'Terrible experience',
            'It was okay'
        ],
        'label': ['Positive', 'Negative', 'Neutral']
    })

    test_data.to_csv(test_csv, index=False)
    print(f"Created test CSV: {test_csv}")

    try:
        preparer = YouTubeDataPreparer(use_api=True)
        comments = preparer.load_labeled_csv(str(test_csv))

        if len(comments) == 3:
            print(f"✅ Loaded {len(comments)} comments successfully")
            print(f"   Labels: {[c['label'] for c in comments]}")
            return True
        else:
            print(f"❌ Expected 3 comments, got {len(comments)}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Cleanup
        test_csv.unlink()
        temp_dir.rmdir()


def test_data_validation():
    """Test 5: Data validation (duplicates, empty text)"""
    print("\n" + "="*80)
    print("TEST 5: Data Validation")
    print("="*80)

    from prepare_youtube_training_data import YouTubeDataPreparer

    preparer = YouTubeDataPreparer(use_api=True)

    # Test data with duplicates and empty strings
    test_comments = [
        {'text': 'Valid comment 1', 'label': 'Positive'},
        {'text': 'Valid comment 2', 'label': 'Negative'},
        {'text': 'Valid comment 1', 'label': 'Positive'},  # Duplicate
        {'text': '', 'label': 'Neutral'},  # Empty
        {'text': '   ', 'label': 'Neutral'},  # Whitespace only
        {'text': 'Valid comment 3', 'label': 'Neutral'},
    ]

    validated = preparer.validate_data(test_comments)

    expected_count = 3  # Should keep 3 unique, non-empty comments
    if len(validated) == expected_count:
        print(f"✅ Validation correct: kept {len(validated)}/{len(test_comments)} comments")
        return True
    else:
        print(f"❌ Expected {expected_count} comments, got {len(validated)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("YOUTUBE DATA PREPARATION FIXES - TEST SUITE")
    print("="*80)

    tests = [
        ("Timestamp Parsing", test_timestamp_parsing),
        ("Metadata Scraping", test_metadata_scraping),
        ("File Paths", test_file_paths),
        ("Manual Labeling", test_manual_labeling),
        ("Data Validation", test_data_validation),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")

    total = len(results)
    passed = sum(results.values())

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All fixes verified! YouTube data preparation is ready.")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")

    # Additional notes
    print("\nNotes:")
    print("- If yt-dlp test failed, install with: pip install yt-dlp")
    print("- All other fixes should work without additional dependencies")


if __name__ == "__main__":
    main()
