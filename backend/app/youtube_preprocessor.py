import re
import string

import emoji
from nltk.tokenize import word_tokenize

# Ensure deterministic language detection when langdetect is used.
# Without this, results may vary run-to-run, which is a thesis-grade reproducibility risk.
try:  # pragma: no cover - depends on optional runtime environment
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
except Exception:
    pass


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

    def _count_tokens(self, text):
        try:
            return len(word_tokenize(text))
        except Exception:
            return len(text.split())

    def _extract_text_features(self, text):
        text = text or ""
        non_space_chars = max(1, sum(1 for char in text if not char.isspace()))
        alpha_chars = sum(1 for char in text if char.isalpha())
        uppercase_chars = sum(1 for char in text if char.isalpha() and char.isupper())
        emoji_count = sum(1 for char in text if emoji.is_emoji(char))
        punctuation_count = sum(1 for char in text if char in string.punctuation)

        return {
            'char_length': len(text),
            'token_length': self._count_tokens(text),
            'emoji_count': emoji_count,
            'emoji_density': round(emoji_count / non_space_chars, 4),
            'uppercase_ratio': round(uppercase_chars / max(alpha_chars, 1), 4),
            'punctuation_ratio': round(punctuation_count / non_space_chars, 4),
            'has_elongation': bool(re.search(r'(.)\1{2,}', text)),
        }

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

    def _apply_common_cleanup(self, text, emoji_mode='convert'):
        text = self.convert_emojis(text, mode=emoji_mode)
        text = self.remove_timestamps(text)
        text = self.remove_channel_mentions(text)
        text = re.sub(r'\[[^]]*\]', ' ', text)
        text = re.sub(r'((http\S+)|(www\.))', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def preprocess_for_classical(self, text, emoji_mode='convert', min_words=3):
        text = self._apply_common_cleanup(text, emoji_mode=emoji_mode)
        text = text.lower()
        text = self.normalize_elongated_words(text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return None
        if not self.filter_short_comments(text, min_words):
            return None
        return text

    def preprocess_for_transformer(self, text, emoji_mode='convert', min_words=3):
        text = self._apply_common_cleanup(text, emoji_mode=emoji_mode)
        if not text:
            return None
        if not self.filter_short_comments(text, min_words):
            return None
        return text

    def preprocess_youtube_comment(self, text, emoji_mode='convert',
                                   check_spam=True, check_lang=True,
                                   min_words=3, profile='classical'):
        metadata = {
            'is_spam': False,
            'language': 'en',
            'filtered': False,
            'filter_reason': None,
            'processing_profile': profile,
            'text_features': self._extract_text_features(text),
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

        selected_profile = (profile or 'classical').lower().strip()
        if selected_profile == 'transformer':
            processed_text = self.preprocess_for_transformer(
                text,
                emoji_mode=emoji_mode,
                min_words=min_words,
            )
        else:
            selected_profile = 'classical'
            metadata['processing_profile'] = selected_profile
            processed_text = self.preprocess_for_classical(
                text,
                emoji_mode=emoji_mode,
                min_words=min_words,
            )

        if processed_text is None:
            metadata['filtered'] = True
            metadata['filter_reason'] = 'too_short'
            return None, metadata

        return processed_text.strip(), metadata

    def batch_preprocess(self, comments, profile='classical', **kwargs):
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
            processed_text, metadata = self.preprocess_youtube_comment(
                text,
                profile=profile,
                **kwargs,
            )

            if metadata['filtered']:
                if metadata['filter_reason'] == 'spam':
                    stats['filtered_spam'] += 1
                elif metadata['filter_reason'] == 'language':
                    stats['filtered_language'] += 1
                elif metadata['filter_reason'] == 'too_short':
                    stats['filtered_short'] += 1
            else:
                comment_copy = comment.copy()
                classical_text, _ = self.preprocess_youtube_comment(
                    text,
                    profile='classical',
                    check_spam=False,
                    check_lang=False,
                    min_words=1,
                    emoji_mode=kwargs.get('emoji_mode', 'convert'),
                )
                transformer_text, _ = self.preprocess_youtube_comment(
                    text,
                    profile='transformer',
                    check_spam=False,
                    check_lang=False,
                    min_words=1,
                    emoji_mode=kwargs.get('emoji_mode', 'convert'),
                )
                comment_copy['processed_text'] = processed_text
                comment_copy['processed_text_classical'] = classical_text or processed_text
                comment_copy['processed_text_transformer'] = transformer_text or processed_text
                comment_copy['metadata'] = metadata
                processed.append(comment_copy)
                stats['processed'] += 1

        return processed, stats
