"""
YouTube Training Data Preparation Script

Fetches YouTube comments and prepares them for deep learning training.
Supports both labeled and unlabeled data extraction.

Usage:
    # Fetch comments from a single video
    python prepare_youtube_training_data.py --video "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --max_comments 500

    # Fetch from multiple videos (from file)
    python prepare_youtube_training_data.py --video_list videos.txt --max_comments 200

    # Export labeled data for training
    python prepare_youtube_training_data.py --video_list videos.txt --output training_data.csv --label_method auto

Author: Master's Thesis
Version: 1.0
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from app.youtube_fetcher import YouTubeFetcher
from app.youtube_scraper import YouTubeScraper
from app.youtube_preprocessor import YouTubePreprocessor
from app.sentiment_engines import get_sentiment_engine


class YouTubeDataPreparer:
    """
    Prepares YouTube comment data for training sentiment models.

    Features:
    - Fetch comments from multiple videos
    - Automatic labeling using LogReg (trained model)
    - Quality filtering (spam, language, length)
    - Train/val/test splitting
    - Export to CSV format for training
    """

    def __init__(self, use_api=True):
        """
        Args:
            use_api (bool): Use YouTube API if True, scraper if False
        """
        self.use_api = use_api

        # Initialize fetcher/scraper
        if use_api and os.getenv("YOUTUBE_API_KEY"):
            self.fetcher = YouTubeFetcher()
            print("✅ Using YouTube API")
        else:
            self.fetcher = YouTubeScraper()
            print("✅ Using YouTube Scraper (no API key)")

        # Initialize preprocessor
        self.preprocessor = YouTubePreprocessor()

        # Initialize labeler lazily for auto-labeling
        self.labeler = None

    def fetch_comments_from_video(self, video_url, max_comments=500):
        """
        Fetch comments from a single video

        Args:
            video_url (str): YouTube video URL or ID
            max_comments (int): Maximum number of comments to fetch

        Returns:
            list: List of comment dictionaries
        """
        video_id = self.fetcher.extract_video_id(video_url)
        print(f"\n📹 Fetching comments from video: {video_id}")

        try:
            comments = self.fetcher.fetch_comments(video_url, max_results=max_comments)
            print(f"   ✅ Fetched {len(comments)} comments")
            return comments
        except Exception as e:
            print(f"   ❌ Error fetching comments: {e}")
            return []

    def fetch_comments_from_list(self, video_list_file, max_comments_per_video=200):
        """
        Fetch comments from multiple videos listed in a file

        Args:
            video_list_file (str): Path to text file with video URLs (one per line)
            max_comments_per_video (int): Max comments per video

        Returns:
            list: Combined list of all comments
        """
        video_list_path = Path(video_list_file)

        if not video_list_path.exists():
            raise FileNotFoundError(f"Video list file not found: {video_list_file}")

        # Read video URLs
        with open(video_list_path, 'r') as f:
            video_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        print(f"\n📋 Loading comments from {len(video_urls)} videos...")

        all_comments = []

        for idx, video_url in enumerate(video_urls, 1):
            print(f"\n[{idx}/{len(video_urls)}] Processing: {video_url}")
            comments = self.fetch_comments_from_video(video_url, max_comments_per_video)
            all_comments.extend(comments)

        print(f"\n✅ Total comments fetched: {len(all_comments)}")
        return all_comments

    def preprocess_comments(self, comments, filter_spam=True, filter_language=True, min_length=10):
        """
        Preprocess and filter comments

        Args:
            comments (list): Raw comments
            filter_spam (bool): Remove spam comments
            filter_language (bool): Keep only English comments
            min_length (int): Minimum comment length in characters

        Returns:
            list: Preprocessed comments
        """
        print(f"\n🔧 Preprocessing {len(comments)} comments...")

        processed = []
        stats = {
            'spam_filtered': 0,
            'language_filtered': 0,
            'too_short': 0,
            'kept': 0
        }

        for comment in comments:
            # Preprocess
            text, metadata = self.preprocessor.preprocess_youtube_comment(
                comment['text'],
                emoji_mode='convert',
                check_spam=filter_spam,
                check_lang=filter_language
            )

            # Check filters
            if metadata['filtered']:
                reason = metadata['filter_reason']
                if 'spam' in reason.lower():
                    stats['spam_filtered'] += 1
                elif 'language' in reason.lower():
                    stats['language_filtered'] += 1
                continue

            # Check length
            if len(text) < min_length:
                stats['too_short'] += 1
                continue

            # Keep comment
            processed.append({
                'text': text,
                'original_text': comment['text'],
                'author': comment.get('author', 'Unknown'),
                'likes': comment.get('likes', 0),
                'published_at': comment.get('published_at', ''),
                'video_id': comment.get('video_id', '')
            })
            stats['kept'] += 1

        # Print statistics
        print(f"   ✅ Preprocessing complete:")
        print(f"      Kept:             {stats['kept']}")
        print(f"      Spam filtered:    {stats['spam_filtered']}")
        print(f"      Language filtered: {stats['language_filtered']}")
        print(f"      Too short:        {stats['too_short']}")

        return processed

    def auto_label_comments(self, comments, confidence_threshold=0.6):
        """
        Automatically label comments using LogReg

        Only keeps comments with high confidence (low entropy).

        Args:
            comments (list): Preprocessed comments
            confidence_threshold (float): Minimum confidence to keep (0-1)

        Returns:
            list: Comments with labels
        """
        print(f"\n🏷️  Auto-labeling {len(comments)} comments with LogReg...")

        if self.labeler is None:
            try:
                self.labeler = get_sentiment_engine('logreg')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "LogReg model files not found. Train the model first with "
                    "`python train_logreg_youtube.py`."
                ) from exc

        labeled = []
        low_confidence = 0

        for comment in comments:
            # Analyze sentiment
            result = self.labeler.analyze(comment['text'])

            # Calculate confidence (1 - entropy)
            from app.analysis_utils import confidence_from_probs
            confidence = confidence_from_probs(result.probs)

            if confidence >= confidence_threshold:
                labeled.append({
                    **comment,
                    'label': result.label,
                    'confidence': confidence,
                    'sentiment_score': result.score
                })
            else:
                low_confidence += 1

        print(f"   ✅ Labeled {len(labeled)} comments (removed {low_confidence} low-confidence)")

        # Print label distribution
        from collections import Counter
        label_counts = Counter([c['label'] for c in labeled])
        print(f"   📊 Label distribution:")
        for label, count in label_counts.items():
            print(f"      {label:10s}: {count:4d} ({count/len(labeled)*100:.1f}%)")

        return labeled

    def load_labeled_csv(self, csv_path):
        """
        Load pre-labeled data from CSV

        Args:
            csv_path (str): Path to CSV file with 'text' and 'label' columns

        Returns:
            list: Comments with labels in standard format
        """
        print(f"\n📄 Loading labeled data from: {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must have 'text' and 'label' columns")

        # Normalize labels
        from app.sentiment_engines import normalize_label
        df['label'] = df['label'].apply(lambda x: normalize_label(str(x)))

        # Convert to comment format
        comments = []
        for _, row in df.iterrows():
            comments.append({
                'text': row['text'],
                'label': row['label'],
                'original_text': row['text'],
                'author': 'Manual',
                'likes': 0,
                'published_at': '',
                'video_id': 'manual',
                'confidence': 1.0,  # High confidence for manual labels
                'sentiment_score': 0.0
            })

        print(f"   ✅ Loaded {len(comments)} pre-labeled comments")

        # Print label distribution
        from collections import Counter
        label_counts = Counter([c['label'] for c in comments])
        print(f"   📊 Label distribution:")
        for label, count in label_counts.items():
            print(f"      {label:10s}: {count:4d} ({count/len(comments)*100:.1f}%)")

        return comments

    def validate_data(self, comments):
        """
        Validate data before export - remove duplicates, empty texts, check balance

        Args:
            comments (list): Comments to validate

        Returns:
            list: Validated comments
        """
        print(f"\n🔍 Validating {len(comments)} comments...")

        stats = {
            'original_count': len(comments),
            'duplicates_removed': 0,
            'empty_removed': 0,
            'kept': 0
        }

        # Track seen texts
        seen_texts = set()
        validated = []

        for comment in comments:
            text = comment['text'].strip()

            # Check empty
            if not text or len(text) == 0:
                stats['empty_removed'] += 1
                continue

            # Check duplicates
            if text in seen_texts:
                stats['duplicates_removed'] += 1
                continue

            seen_texts.add(text)
            validated.append(comment)
            stats['kept'] += 1

        # Check label distribution
        from collections import Counter
        label_counts = Counter([c['label'] for c in validated])
        total = len(validated)

        print(f"   ✅ Validation complete:")
        print(f"      Kept:       {stats['kept']}")
        print(f"      Duplicates: {stats['duplicates_removed']}")
        print(f"      Empty:      {stats['empty_removed']}")
        print(f"\n   📊 Final label distribution:")

        for label, count in label_counts.items():
            pct = count / total * 100
            status = "⚠️ " if pct > 80 else "✅"
            print(f"      {status} {label:10s}: {count:4d} ({pct:.1f}%)")

        # Warn if imbalanced
        max_pct = max(label_counts.values()) / total * 100
        if max_pct > 80:
            print(f"\n   ⚠️  Warning: Dataset is imbalanced ({max_pct:.1f}% in one class)")
            print(f"      Consider collecting more diverse samples for better model training.")

        return validated

    def split_train_val_test(self, comments, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        Split data into train/val/test sets with stratification

        Args:
            comments (list): Labeled comments
            train_ratio (float): Training set proportion
            val_ratio (float): Validation set proportion
            test_ratio (float): Test set proportion
            random_seed (int): Random seed

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        print(f"\n✂️  Splitting data (train={train_ratio}, val={val_ratio}, test={test_ratio})...")

        # Convert to DataFrame
        df = pd.DataFrame(comments)

        # Stratified split
        train_val, test = train_test_split(
            df, test_size=test_ratio, random_state=random_seed, stratify=df['label']
        )

        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val, test_size=val_ratio_adjusted, random_state=random_seed, stratify=train_val['label']
        )

        print(f"   ✅ Split complete:")
        print(f"      Train: {len(train)} samples")
        print(f"      Val:   {len(val)} samples")
        print(f"      Test:  {len(test)} samples")

        return train, val, test

    def export_to_csv(self, train_df, val_df, test_df, output_dir='./data/youtube_training'):
        """
        Export datasets to CSV files

        Args:
            train_df (DataFrame): Training data
            val_df (DataFrame): Validation data
            test_df (DataFrame): Test data
            output_dir (str): Output directory

        Returns:
            dict: Paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save files
        train_path = output_path / f'train_{timestamp}.csv'
        val_path = output_path / f'val_{timestamp}.csv'
        test_path = output_path / f'test_{timestamp}.csv'

        # Save only essential columns
        columns = ['text', 'label']
        train_df[columns].to_csv(train_path, index=False)
        val_df[columns].to_csv(val_path, index=False)
        test_df[columns].to_csv(test_path, index=False)

        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'label_distribution': {
                'train': train_df['label'].value_counts().to_dict(),
                'val': val_df['label'].value_counts().to_dict(),
                'test': test_df['label'].value_counts().to_dict()
            }
        }

        metadata_path = output_path / f'metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n💾 Data exported to: {output_path}")
        print(f"   Train: {train_path.name}")
        print(f"   Val:   {val_path.name}")
        print(f"   Test:  {test_path.name}")
        print(f"   Metadata: {metadata_path.name}")

        return {
            'train': str(train_path),
            'val': str(val_path),
            'test': str(test_path),
            'metadata': str(metadata_path)
        }


def main():
    """Main pipeline"""
    parser = argparse.ArgumentParser(
        description='Prepare YouTube comments for training sentiment models'
    )

    # Input sources
    parser.add_argument('--video', type=str, help='Single video URL or ID')
    parser.add_argument('--video_list', type=str, help='File with video URLs (one per line)')
    parser.add_argument('--max_comments', type=int, default=500, help='Max comments per video')

    # Preprocessing
    parser.add_argument('--filter_spam', action='store_true', default=True, help='Filter spam')
    parser.add_argument('--filter_language', action='store_true', default=True, help='Filter non-English')
    parser.add_argument('--min_length', type=int, default=10, help='Min comment length')

    # Labeling
    parser.add_argument('--label_method', choices=['auto', 'manual'], default='auto',
                       help='Labeling method (auto=LogReg, manual=requires labels)')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Min confidence for auto-labeling (0-1)')
    parser.add_argument('--labeled_csv', type=str,
                       help='Pre-labeled CSV file (text,label columns)')
    parser.add_argument('--merge_mode', choices=['replace', 'append'], default='replace',
                       help='How to handle labeled data: replace (only use labeled) or append (mix with auto-labeled)')

    # Splitting
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--random_seed', type=int, default=42)

    # Output
    parser.add_argument('--output_dir', type=str, default='./data/youtube_training',
                       help='Output directory')
    parser.add_argument('--use_api', action='store_true', default=True,
                       help='Use YouTube API (requires API key)')

    args = parser.parse_args()

    # Validate inputs
    if not args.video and not args.video_list and not args.labeled_csv:
        parser.error("Either --video, --video_list, or --labeled_csv must be provided")

    print("="*80)
    print("YOUTUBE TRAINING DATA PREPARATION")
    print("="*80)

    # Initialize preparer
    preparer = YouTubeDataPreparer(use_api=args.use_api)

    # 1. Fetch comments (skip if only using labeled CSV)
    comments = []
    if args.video or args.video_list:
        if args.video:
            comments = preparer.fetch_comments_from_video(args.video, args.max_comments)
        else:
            comments = preparer.fetch_comments_from_list(args.video_list, args.max_comments)

        if not comments:
            print("\n❌ No comments fetched. Exiting.")
            return

        # 2. Preprocess
        comments = preparer.preprocess_comments(
            comments,
            filter_spam=args.filter_spam,
            filter_language=args.filter_language,
            min_length=args.min_length
        )

        if not comments:
            print("\n❌ No comments passed preprocessing. Exiting.")
            return

    # 3. Label
    if args.labeled_csv:
        # Load pre-labeled data
        labeled_comments = preparer.load_labeled_csv(args.labeled_csv)

        if args.merge_mode == 'append' and args.label_method == 'auto':
            # Mix manual labels with auto-labeled data
            auto_labeled = preparer.auto_label_comments(comments, args.confidence_threshold)
            comments = labeled_comments + auto_labeled
            print(f"\n🔀 Merged {len(labeled_comments)} manual + {len(auto_labeled)} auto-labeled = {len(comments)} total")
        else:
            # Use only manual labels
            comments = labeled_comments
            print(f"\n📊 Using {len(comments)} pre-labeled comments only")
    elif args.label_method == 'auto':
        # Auto-label with LogReg
        comments = preparer.auto_label_comments(comments, args.confidence_threshold)
    else:
        print("\n⚠️  No labeling method specified. Use --label_method auto or --labeled_csv")
        return

    if not comments:
        print("\n❌ No comments passed labeling. Exiting.")
        return

    # 3.5. Validate data (remove duplicates, empty text, check balance)
    comments = preparer.validate_data(comments)

    if not comments:
        print("\n❌ No comments passed validation. Exiting.")
        return

    # 4. Split
    train_df, val_df, test_df = preparer.split_train_val_test(
        comments,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )

    # 5. Export
    paths = preparer.export_to_csv(train_df, val_df, test_df, output_dir=args.output_dir)

    print("\n" + "="*80)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print(f"1. Review the data in: {args.output_dir}")
    print(f"2. Train a model:")
    print(f"   python research/train_hybrid_dl.py \\")
    print(f"       --train_csv {paths['train']} \\")
    print(f"       --val_csv {paths['val']} \\")
    print(f"       --test_csv {paths['test']}")
    print()


if __name__ == "__main__":
    main()
