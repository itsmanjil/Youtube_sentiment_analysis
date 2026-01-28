# YouTube Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Django](https://img.shields.io/badge/Django-5.0+-green.svg)
![React](https://img.shields.io/badge/React-19-blue.svg)
![Vite](https://img.shields.io/badge/Vite-7-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A full-stack sentiment analysis platform that analyzes YouTube video comments to surface audience sentiment and engagement insights. Built with Django REST Framework, React 19, and Vite.

## Overview

This application provides comprehensive sentiment analysis of YouTube video comments, helping content creators, marketers, and researchers understand audience reactions and engagement patterns. The system fetches comments using either the official YouTube API or a fallback scraper, processes them through advanced NLP pipelines, and presents the results through an intuitive React dashboard.

**Key Capabilities:**
- Analyze up to 1000 comments per video
- Support for multiple sentiment models (LogReg, Linear SVM, TF-IDF, Ensemble, Meta-Learner, Hybrid-DL, BERT/Transformer)
- Real-time spam and language filtering
- Like-weighted sentiment analysis
- Word frequency analysis for word clouds
- Aspect-based sentiment analysis
- Confidence intervals with bootstrap estimation
- Sentiment timeline visualization
- JWT-authenticated user system
- Analysis history and comparison
- Research tools for thesis-grade evaluation

## Demo

### Sample Analysis Flow
1. User enters a YouTube video URL
2. System fetches comments via API or scraper
3. Comments are preprocessed (spam filtering, language detection, emoji handling)
4. Sentiment analysis is performed using selected model
5. Results are displayed with:
   - Sentiment distribution pie chart
   - Like-weighted sentiment breakdown
   - Top positive/negative words
   - Filter statistics
6. Analysis is saved to user's history

### Example Output
```json
{
  "sentiment_data": {
    "Positive": 150,
    "Negative": 30,
    "Neutral": 20
  },
  "sentiment_ratio": {
    "positive_percent": 75.0,
    "negative_percent": 15.0,
    "neutral_percent": 10.0
  },
  "total_analyzed": 200,
  "model_used": "LOGREG"
}
```

## Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#4-environment-variables)
- [API Endpoints](#api-endpoints)
- [Usage](#usage)
- [Testing](#testing)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Sentiment Models](#sentiment-models)
- [Use Cases](#use-cases)
- [Authentication](#authentication)
- [Security](#security)
- [Performance](#performance)
- [Database Models](#database-models)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Credits](#credits)
- [Support](#support)
- [License](#license)

## Features

### Data Collection
- **Dual fetching modes**: YouTube Data API v3 (fast, reliable) or scraper (unlimited, no API key)
- **Video metadata**: title, channel, views, likes, comment count
- **Comment data**: text, author, likes, timestamps, reply status

### Sentiment Analysis
- **LogReg (TF-IDF)**: Trainable baseline tuned for YouTube comments
- **Linear SVM (TF-IDF)**: Strong linear model trained on YouTube data
- **TF-IDF**: Naive Bayes baseline (fastest)
- **Ensemble**: Weighted soft-voting with configurable weights
- **Meta-Learner**: Stacked ensemble that learns combination rules
- **Hybrid-DL**: CNN-BiLSTM-Attention research model (requires PyTorch)
- **BERT/Transformer**: Transformer-based model (requires transformers library, best accuracy)
- **Confidence scores**: Compound sentiment scores for each comment
- **Uncertainty scoring**: Entropy-based confidence for each prediction

### Preprocessing
- **Spam detection**: Filter promotional and bot comments
- **Language filtering**: English-only option
- **Emoji handling**: Remove, convert to text, or keep
- **Text normalization**: Contractions, negations, stopwords, elongated words

### Analytics
- **Sentiment ratios**: Positive/Negative/Neutral percentages
- **Like-weighted analysis**: Highlight influential comments by likes
- **Word clouds**: Top 50 positive and negative words
- **Top comments**: Most engaging comments by likes
- **Filter statistics**: Track spam, language, and short comment filtering
- **Confidence intervals**: Bootstrap CI for sentiment ratios
- **Aspect sentiment**: Keyword-based aspect extraction with sentiment breakdown
- **Sentiment timeline**: Hourly sentiment distribution

### Frontend
- **React dashboard**: Material design UI
- **User authentication**: JWT-based login/register
- **Data visualization**: Charts with Recharts
- **PDF export**: Analysis reports
- **Analysis history**: View past analyses

## Project Structure

```
Youtube_sentiment_analysis/
├── backend/                    # Django REST API
│   ├── app/                   # Main YouTube analysis app
│   │   ├── models.py         # YouTubeVideo, YouTubeComment, YouTubeAnalysis
│   │   ├── views.py          # API endpoints
│   │   ├── youtube_fetcher.py    # YouTube API client
│   │   ├── youtube_scraper.py    # Alternative scraper
│   │   ├── youtube_preprocessor.py  # Text cleaning & filtering
│   │   ├── aspect_mining.py      # Aspect-based sentiment extraction
│   │   └── deep_models.py        # Deep learning model definitions
│   ├── src/                  # Modular sentiment engine package
│   │   ├── sentiment/        # Sentiment analysis engines
│   │   │   ├── factory.py    # Engine factory with lazy loading
│   │   │   └── engines/      # LogReg, SVM, TF-IDF, Ensemble, Meta-Learner, Hybrid-DL, Transformer
│   │   └── utils/            # Analysis utilities (confidence, bootstrap CI)
│   ├── research/             # Thesis-grade research tools
│   │   ├── architectures/    # CNN-BiLSTM-Attention, BERT classifier
│   │   ├── training/         # Trainer, evaluator, callbacks
│   │   ├── evaluation/       # Ablation studies, statistical tests
│   │   ├── explainability/   # LIME, SHAP, attention explainer
│   │   ├── visualization/    # Training curves, confusion matrices
│   │   ├── experiment_runner.py   # Full evaluation pipeline
│   │   ├── optimize_ensemble.py   # PSO weight optimization
│   │   └── run_thesis_pipeline.py # Complete thesis experiments
│   ├── app_api/              # Authentication API
│   ├── users/                # User management
│   ├── core/                 # Django settings & config
│   ├── files/                # Data files (contractions, negations)
│   ├── models/               # Trained model artifacts
│   ├── test_youtube.py       # Integration tests
│   ├── manage.py
│   └── Pipfile               # Python dependencies (Python 3.11)
├── frontend/                  # React 19 + Vite application
│   ├── src/
│   │   ├── Views/
│   │   │   ├── Pages/        # Dashboard, Search, Report, Monitoring, Tables
│   │   │   └── Account Pages/ # Signin, Register, Profile, EditProfile
│   │   ├── Components/       # Reusable components
│   │   ├── context/          # Auth context
│   │   └── axios.js          # API client
│   ├── vite.config.js        # Vite configuration
│   └── package.json
└── README.md
```

## Quick Start

### Prerequisites

**Backend:**
- Python 3.11+
- pip or pipenv
- PostgreSQL or SQLite (SQLite by default)
- PyTorch (optional, for Hybrid-DL and Transformer models)

**Frontend:**
- Node.js 18+
- npm or yarn

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/itsmanjil/Youtube_sentiment_analysis.git
cd Youtube_sentiment_analysis
```

#### 2. Backend Setup
```bash
cd backend

# Option A: Using pipenv (recommended)
pipenv install
pipenv shell

# Option B: Using pip and venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt  # If requirements.txt exists
# OR install from Pipfile: pip install django djangorestframework django-cors-headers ...

# Create .env file
cp .env.example .env
# Edit .env and set:
#   - SECRET_KEY (Django secret key)
#   - YOUTUBE_API_KEY (optional - get from Google Cloud Console)

# Download NLTK data
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Run migrations
python3 manage.py migrate

# Create superuser (admin account)
python3 manage.py createsuperuser

# Run tests (optional)
python3 test_youtube.py

# Start development server
python3 manage.py runserver
```

Backend will run on http://localhost:8000

#### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server (Vite)
npm run dev
```

Frontend will run on http://localhost:5173 (Vite default)

### 4. Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Django Configuration
SECRET_KEY=your-secret-key-here-generate-a-random-string
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (SQLite by default, uncomment for PostgreSQL)
# DATABASE_URL=postgresql://user:password@localhost:5432/youtube_sentiment

# YouTube API (Optional - scraper works without it)
YOUTUBE_API_KEY=your-youtube-api-key-here

# CORS (for frontend - Vite default port)
CORS_ALLOWED_ORIGINS=http://localhost:5173
```

**Generate a Secret Key:**
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 5. Get a YouTube API Key (Optional)
The scraper works without an API key, but the official API is faster and more reliable:

1. Go to https://console.cloud.google.com/
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create credentials → API Key
5. Copy the key to your `.env` file as `YOUTUBE_API_KEY`

**Free tier:** 10,000 units/day (approximately 100 video analyses)

## Usage

### Analyze a YouTube Video

With API (recommended):
```bash
POST /api/youtube/analyze/
{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "max_comments": 200,
  "use_api": true,
  "sentiment_model": "logreg"
}
```

Without API (scraper mode):
```bash
POST /api/youtube/analyze/
{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "max_comments": 100,
  "use_api": false,
  "sentiment_model": "logreg"
}
```

### Example Response
```json
{
  "msg": "Analysis complete",
  "video": {
    "id": "dQw4w9WgXcQ",
    "title": "Rick Astley - Never Gonna Give You Up",
    "channel": "Rick Astley",
    "view_count": 1234567890,
    "like_count": 12000000
  },
  "sentiment_data": {
    "Positive": 150,
    "Negative": 30,
    "Neutral": 20
  },
  "sentiment_ratio": {
    "positive_percent": 75.0,
    "negative_percent": 15.0,
    "neutral_percent": 10.0
  },
  "like_weighted_sentiment": [],
  "top_words_positive": [
    {"word": "love", "count": 45},
    {"word": "amazing", "count": 32}
  ]
}
```

## Testing

### Backend Tests
```bash
cd backend
python test_youtube.py
```

Expected output:
```
Video ID Extraction
Comment Preprocessing
LogReg Sentiment
YouTube API (or Scraper)
All tests passed
```

### Frontend Tests
```bash
cd frontend
npm test
```

Tests run using Vitest with React Testing Library.

## Research & Experiments

The project includes comprehensive research tools for thesis-grade experiments:

```bash
# Run full experiment evaluation
python backend/research/experiment_runner.py --data path/to/labeled.csv --use-full

# Optimize ensemble weights using PSO
python backend/research/optimize_ensemble.py --data path/to/labeled.csv --use-full

# Evaluate fuzzy ensemble (uncertainty-aware)
python backend/research/fuzzy_experiment_runner.py --data path/to/labeled.csv --use-full

# Grid search fuzzy configs (CI tuning)
python backend/research/fuzzy_grid_search.py --data path/to/labeled.csv --use-full --limit 5000

# Train hybrid deep learning model
python backend/research/train_hybrid_dl.py --data path/to/labeled.csv

# Train meta-learner (stacked ensemble)
python backend/research/meta_learner.py --data path/to/labeled.csv

# Run complete thesis pipeline
python backend/research/run_thesis_pipeline.py --data path/to/labeled.csv

# Generate visualizations
python backend/research/visualization/generate_all.py
```

### Research Features
- **Ablation studies**: Systematic component analysis
- **Statistical tests**: McNemar's test, paired t-tests
- **Explainability**: LIME, SHAP, attention visualization
- **Training curves**: Loss and accuracy plots
- **Confusion matrices**: Per-model performance breakdown
- **CI tuning**: PSO ensemble weights + fuzzy config grid search

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/users/register/` | Register new user |
| POST | `/api/token/` | Login and get JWT tokens |
| POST | `/api/token/refresh/` | Refresh access token |

### YouTube Analysis
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/youtube/analyze/` | Analyze YouTube video comments | Yes |
| GET | `/api/youtube/analysis/<video_id>/` | Get saved analysis for a video | Yes |
| GET | `/api/youtube/analyses/` | Get all user's analyses (last 20) | Yes |
| GET | `/api/youtube/test/` | Health check endpoint | No |

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_url` | string | required | YouTube video URL or ID |
| `max_comments` | integer | 200 | Number of comments to analyze (1-1000) |
| `use_api` | boolean | true | Use official API (true) or scraper (false) |
| `emoji_mode` | string | "convert" | How to handle emojis: "remove", "convert", "keep" |
| `filter_spam` | boolean | true | Filter spam comments |
| `filter_language` | boolean | true | Filter non-English comments |
| `sentiment_model` | string | "logreg" | Model to use: "logreg", "svm", "tfidf", "ensemble", "meta_learner", "hybrid_dl", "bert" |
| `ensemble_models` | string/list | "logreg,svm,tfidf" | Base models for the ensemble |
| `ensemble_weights` | object/list | null | Weights for ensemble (dict or list, or path to JSON file) |
| `meta_learner_path` | string | null | Path to trained meta-learner model |
| `meta_learner_models` | string/list | null | Base models for meta-learner |
| `bootstrap_samples` | integer | 500 | Bootstrap samples for CI estimation |
| `random_seed` | integer | 42 | Seed for reproducibility |
| `aspect_top_n` | integer | 12 | Number of aspects to return |
| `aspect_min_freq` | integer | 3 | Minimum aspect frequency |
| `confidence_threshold` | float | 0.6 | Threshold for low-confidence ratio |
| `fuzzy_models` | list | null | Fuzzy base models (e.g., ["logreg","svm"]) |
| `fuzzy_mf_type` | string | null | MF type: triangular, trapezoidal, gaussian |
| `fuzzy_defuzz_method` | string | null | Defuzz: centroid, bisector, mom, som, lom |
| `fuzzy_t_norm` | string | null | T-norm: min, product, lukasiewicz |
| `fuzzy_t_conorm` | string | null | T-conorm: max, prob_sum, bounded_sum |
| `fuzzy_alpha_cut` | float | null | Alpha-cut threshold |
| `fuzzy_resolution` | integer | null | Fuzzy resolution (e.g., 100) |
| `model_comparison` | list | null | Optional metrics list for report tables |

## Dataset Preparation (HF YouTube Dataset)

Use the Hugging Face YouTube dataset and generate fixed train/val/test splits:

```bash
python backend/scripts/prepare/prepare_hf_dataset.py
```

This creates:
- `backend/data/train.csv`
- `backend/data/val.csv`
- `backend/data/test.csv`

These splits should be used consistently across classical, ensemble, PSO, and fuzzy experiments.

## Architecture

```
YouTube Video URL
  -> Fetcher/Scraper
     - YouTube API: fast, 10k quota/day
     - Scraper (yt-dlp): unlimited, slower
  -> Preprocessor
     - Emoji handling (remove/convert/keep)
     - Spam detection
     - Language filtering
     - Contraction expansion
     - Negation handling
     - Stopword removal
  -> Sentiment Engine Factory
     - TF-IDF (Naive Bayes)
     - LogReg (default)
     - Linear SVM
     - Ensemble (weighted voting)
     - Meta-Learner (stacked)
     - Hybrid-DL (CNN-BiLSTM-Attention)
     - BERT/Transformer
  -> Analytics
     - Sentiment ratios + confidence intervals
     - Bootstrap CI estimation
     - Aspect-based sentiment
     - Sentiment timeline
     - Like-weighted analysis
     - Word frequency analysis
  -> Database
     - YouTubeVideo
     - YouTubeComment
     - YouTubeAnalysis
```

## Technology Stack

### Backend
- **Django 4.0+**: Web framework
- **Django REST Framework**: RESTful API
- **djangorestframework-simplejwt**: JWT authentication
- **PostgreSQL / SQLite**: Database (SQLite default)
- **python-dotenv**: Environment variables

### YouTube & Data Collection
- **google-api-python-client**: Official YouTube Data API v3
- **youtube-comment-downloader**: Fallback comment scraper
- **yt-dlp**: Video metadata extraction
- **googleapiclient**: API error handling

### NLP & Sentiment Analysis
- **scikit-learn 1.8**: TF-IDF + Logistic Regression + Linear SVM + Meta-Learner
- **torch**: PyTorch for Hybrid-DL (CNN-BiLSTM-Attention)
- **transformers**: BERT and other transformer models (optional)
- **NLTK**: Tokenization, stopwords, wordnet
- **pandas**: Data processing
- **numpy 2.0+**: Numerical computing

### Text Processing
- **emoji**: Emoji detection and conversion
- **langdetect**: Language identification
- **wordcloud**: Word frequency visualization
- **matplotlib**: Plotting

### Frontend
- **React 19**: UI framework
- **Vite 7**: Build tool and dev server
- **React Router 7**: Navigation
- **Axios**: HTTP client
- **jwt-decode**: JWT token handling
- **Recharts 3**: Data visualization
- **Bootstrap 5**: Styling
- **Material Dashboard**: UI theme
- **Vitest**: Testing framework

## Sentiment Models

| Model | Speed | Accuracy | GPU | Best For |
|-------|-------|----------|-----|----------|
| TF-IDF (Naive Bayes) | Fastest | Medium | No | Quick baseline |
| LogReg (TF-IDF) | Fast | Medium-High | No | YouTube comments (default) |
| Linear SVM (TF-IDF) | Fast | High | No | Highest classical accuracy |
| Ensemble | Medium | High | No | Combined classical models |
| Meta-Learner | Medium | High | No | Learned model combination |
| Hybrid-DL | Slow | High | Optional | CNN-BiLSTM-Attention research model |
| BERT/Transformer | Slowest | Highest | Recommended | Best accuracy (requires transformers) |

**Requirements:**
- Classical models (TF-IDF, LogReg, SVM, Ensemble, Meta-Learner): No additional dependencies
- Hybrid-DL: Requires `torch`
- BERT/Transformer: Requires `torch` and `transformers`

Recommendation: Use LogReg or Linear SVM for production; BERT for best accuracy.

## Use Cases

### Product Launch Analysis
Track sentiment on product announcement videos:
- Tech product launches (Apple, Samsung, Tesla)
- Game trailers and announcements
- Movie/TV show previews

### Brand Monitoring
Monitor brand reputation across YouTube:
- Real-time sentiment tracking
- Influencer impact analysis
- Competitor comparison

### Content Creator Insights
Help creators understand their audience:
- Video performance analysis
- Comment sentiment trends
- Identify controversial topics
- Most appreciated content

### Political and News Analysis
Analyze public opinion on current events:
- Debate videos
- News coverage sentiment
- Policy announcement reactions

### Educational Content
Help educators improve courses:
- Identify confusing topics
- Track student satisfaction
- Common questions from comments

## Authentication

The application uses JWT (JSON Web Tokens) for authentication:

### User Registration
```bash
POST /api/users/register/
{
  "user_name": "username",
  "email": "user@example.com",
  "password": "securepassword"
}
```

### User Login
```bash
POST /api/token/
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

Returns:
```json
{
  "access": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

### Using the Token
Include the access token in all API requests:
```bash
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

### Token Refresh
```bash
POST /api/token/refresh/
{
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

The frontend automatically manages tokens using React Context and stores them in localStorage.

## Security
- **API keys**: Stored in `.env` (never commit real keys)
- **JWT authentication**: Secure token-based auth
- **Django security middleware**: CSRF, XSS protection
- **SQL injection protection**: Django ORM
- **Input validation**: All endpoints validate input
- **CORS configuration**: Configured for frontend
- **Password hashing**: Django's built-in password hasher

## Performance

### YouTube API Mode
- Speed: 2-5 seconds for 100 comments
- Quota: about 100 videos/day (free tier)
- Reliability: 99.9%
- Metadata: full (title, views, likes)

### Scraper Mode
- Speed: 5-15 seconds for 100 comments
- Quota: unlimited
- Reliability: 85% (can break)
- Metadata: minimal

### LogReg (TF-IDF) Sentiment
- Speed: fast for typical 100-comment batches on CPU
- Accuracy: medium-high on YouTube comment data
- Memory: moderate (vectorizer + model artifacts)
- GPU: not required

## Contributing

Contributions are welcome! Here's how you can help:

### Getting Started
1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/Youtube_sentiment_analysis.git`
3. Create a feature branch: `git checkout -b feature/AmazingFeature`
4. Make your changes
5. Run tests: `python test_youtube.py` (backend) and `npm test` (frontend)
6. Commit your changes: `git commit -m 'Add AmazingFeature'`
7. Push to the branch: `git push origin feature/AmazingFeature`
8. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide for Python code
- Use ESLint for JavaScript/React code
- Write tests for new features
- Update documentation as needed
- Keep commits atomic and well-described

### Areas for Contribution
- Add new sentiment models
- Improve preprocessing algorithms
- Enhance frontend UI/UX
- Add visualization features
- Write comprehensive tests
- Improve documentation
- Fix bugs and issues

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Credits

### Libraries & Tools
- [YouTube Data API v3](https://developers.google.com/youtube/v3) - Official YouTube API
- [youtube-comment-downloader](https://github.com/egbertbouman/youtube-comment-downloader) - Alternative scraper
- [Django](https://www.djangoproject.com/) - Web framework
- [React](https://reactjs.org/) - Frontend framework
- [Material Dashboard](https://www.creative-tim.com/product/material-dashboard-react) - UI theme
- [emoji](https://github.com/carpedm20/emoji) - Emoji processing
- [langdetect](https://github.com/Mimino666/langdetect) - Language detection
- [NLTK](https://www.nltk.org/) - Natural language toolkit
- [Recharts](https://recharts.org/) - React charting library

## Support
- Run `python test_youtube.py` for diagnostics
- Open an issue for bugs or feature requests

## Roadmap
- [x] YouTube sentiment analysis
- [x] LogReg sentiment engine
- [x] SVM and TF-IDF engines
- [x] Ensemble and Meta-Learner engines
- [x] Hybrid-DL (CNN-BiLSTM-Attention) engine
- [x] BERT/Transformer engine
- [x] Spam detection and language filtering
- [x] Like-weighted sentiment
- [x] React dashboard with Material Design
- [x] JWT authentication
- [x] User analysis history
- [x] Aspect-based sentiment analysis
- [x] Confidence intervals with bootstrap
- [x] Sentiment timeline visualization
- [x] Research tools (LIME, SHAP, ablation)
- [x] Vite build system for frontend
- [ ] Word cloud visualization endpoint
- [ ] Video transcript analysis
- [ ] Multi-language support
- [ ] Real-time monitoring dashboard
- [ ] Export to CSV/Excel
- [ ] Channel-level analytics
- [ ] Comment thread analysis

## Database Models

### YouTubeVideo
Stores video metadata fetched from YouTube:
- `video_id` (PK): 11-character YouTube video ID
- `title`, `description`: Video content
- `channel_name`, `channel_id`: Creator info
- `published_at`: Upload timestamp
- `view_count`, `like_count`, `comment_count`: Engagement metrics
- `thumbnail_url`: Video thumbnail

### YouTubeComment
Stores individual comments with sentiment analysis:
- `video` (FK): Reference to YouTubeVideo
- `comment_id`: YouTube comment ID
- `text`, `author`: Comment content
- `likes`, `published_at`: Engagement data
- `is_reply`: Thread structure
- `sentiment`: Positive/Negative/Neutral
- `sentiment_score`: Compound score (-1 to +1)
- `is_spam`, `language`: Filter metadata

### YouTubeAnalysis
Stores complete sentiment analysis results:
- `user` (FK): User who requested analysis
- `video` (FK): Analyzed video
- `sentiment_data`: Sentiment counts (JSON)
- `like_weighted_sentiment`: Top liked comments (JSON)
- `top_words_positive`, `top_words_negative`: Word frequencies (JSON)
- `total_comments_analyzed`: Analysis scope
- `filtered_spam_count`, `filtered_language_count`: Filter stats
- `analysis_model`: LogReg/SVM/TF-IDF/Ensemble/Hybrid-DL
- `fetched_date`: Analysis timestamp

## Troubleshooting

### NLTK Data Not Found
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### YouTube API Quota Exceeded
Switch to scraper mode:
```json
{
  "video_url": "...",
  "use_api": false
}
```

### Scraper Not Working
The scraper may be blocked by YouTube. Try:
1. Use a different video
2. Wait a few minutes and retry
3. Use the official API with a key

### Import Errors
Ensure all dependencies are installed:
```bash
cd backend
pipenv install  # or pip install -r requirements.txt
```

### CORS Errors
Check `backend/core/settings.py` and ensure:
```python
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite default port
]
```

### Database Migration Issues
```bash
python manage.py makemigrations
python manage.py migrate
```

### Frontend Not Connecting to Backend
Check that:
1. Backend is running on http://localhost:8000
2. Frontend `axios.js` points to correct API URL
3. CORS is configured correctly for http://localhost:5173

---

Built with Django REST Framework, React 19, Vite, and a modular sentiment engine architecture supporting classical ML, ensemble methods, and deep learning models.
