
import re
import emoji
from nltk.tokenize import word_tokenize


class YouTubePreprocessor:

    # Common spam patterns in YouTube comments
    SPAM_PATTERNS = [
        r'(?i)(check out my channel)',
        r'(?i)(subscribe to me)',
        r'(?i)(sub to me)',
        r'(?i)(sub 4 sub)',
        r'(?i)(sub for sub)',
        r'(?i)(follow me)',
        r'(?i)(click here)',
        r'(?i)(click the link)',
        r'(?i)(http[s]?://(?!youtube|youtu\.be))',  # Non-YT links
        r'(?i)(free (money|gift|iphone|robux))',
        r'(?i)(make \$?\d+ (per|a) (day|week|month))',
        r'(?i)(earn money)',
        r'(?i)(watch my)',
        r'(?i)(dm me)',
        r'(\d{4,})',  # Long number sequences (phone numbers)
        r'((.)\2{10,})',  # 10+ repeated characters
        r'(?i)(type amen)',
        r'(?i)(copy and paste)',
    ]

    def __init__(self):
        self.spam_regex = [re.compile(pattern) for pattern in self.SPAM_PATTERNS]

    def is_spam(self, text):
        for pattern in self.spam_regex:
            if pattern.search(text):
                return True
        return False

    def detect_language(self, text):
        try:
            from langdetect import detect, LangDetectException
            return detect(text)
        except (ImportError, Exception):
            # If langdetect not available or fails, assume English
            return 'en'

    def convert_emojis(self, text, mode='remove'):
        if mode == 'remove':
            return emoji.replace_emoji(text, replace='')
        elif mode == 'convert':
            # Convert emoji to text like :smiling_face:
            text = emoji.demojize(text)
            # Remove colons and underscores
            text = re.sub(r':(\w+):', r' \1 ', text)
            text = text.replace('_', ' ')
            return text
        else:  # keep
            return text

    def remove_timestamps(self, text):
        # Matches MM:SS or HH:MM:SS
        return re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', '', text)

    def normalize_elongated_words(self, text):
        # Replace 3+ repeated chars with single char
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def remove_channel_mentions(self, text):
        return re.sub(r'@[\w\s-]+', '', text)

    def filter_short_comments(self, text, min_words=3):
        try:
            words = word_tokenize(text)
            return len(words) >= min_words
        except:
            # Fallback to simple split if tokenization fails
            return len(text.split()) >= min_words

    def preprocess_youtube_comment(self, text, emoji_mode='convert',
                                   check_spam=True, check_lang=True,
                                   min_words=3):
        metadata = {
            'is_spam': False,
            'language': 'en',
            'filtered': False,
            'filter_reason': None
        }

        # Spam detection (check before processing)
        if check_spam and self.is_spam(text):
            metadata['is_spam'] = True
            metadata['filtered'] = True
            metadata['filter_reason'] = 'spam'
            return None, metadata

        # Language detection (before heavy preprocessing)
        if check_lang:
            try:
                lang = self.detect_language(text)
                metadata['language'] = lang
                if lang != 'en':
                    metadata['filtered'] = True
                    metadata['filter_reason'] = 'language'
                    return None, metadata
            except:
                # If language detection fails, assume English
                metadata['language'] = 'en'

        # Emoji handling
        text = self.convert_emojis(text, mode=emoji_mode)

        # YouTube-specific cleaning
        text = self.remove_timestamps(text)
        text = self.remove_channel_mentions(text)

        # Standard cleaning
        text = text.lower()
        text = self.normalize_elongated_words(text)
        text = re.sub(r'\[[^]]*\]', '', text)  # Remove square brackets
        text = re.sub(r'((http\S+)|(www\.))', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
        text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove single chars
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        # Filter short comments
        if not self.filter_short_comments(text.strip(), min_words):
            metadata['filtered'] = True
            metadata['filter_reason'] = 'too_short'
            return None, metadata

        return text.strip(), metadata

    def batch_preprocess(self, comments, **kwargs):
        processed = []
        stats = {
            'total': len(comments),
            'processed': 0,
            'filtered_spam': 0,
            'filtered_language': 0,
            'filtered_short': 0
        }

        for comment in comments:
            text = comment.get('text', '')
            processed_text, metadata = self.preprocess_youtube_comment(text, **kwargs)

            if metadata['filtered']:
                if metadata['filter_reason'] == 'spam':
                    stats['filtered_spam'] += 1
                elif metadata['filter_reason'] == 'language':
                    stats['filtered_language'] += 1
                elif metadata['filter_reason'] == 'too_short':
                    stats['filtered_short'] += 1
            else:
                comment_copy = comment.copy()
                comment_copy['processed_text'] = processed_text
                comment_copy['metadata'] = metadata
                processed.append(comment_copy)
                stats['processed'] += 1

        return processed, stats
