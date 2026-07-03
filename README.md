# YouTube Sentiment Analysis

A full-stack app for analyzing YouTube video comments with Django REST Framework on the backend and React/Vite on the frontend. It supports authenticated analysis history, multiple sentiment engines, report generation, and a thesis-oriented research layer under `backend/research/`.

## Stack

- Backend: Django, Django REST Framework, SimpleJWT, scikit-learn, NLTK
- Frontend: React 19, Vite 7, React Router, Recharts, Vitest
- Data: SQLite by default
- CI: GitHub Actions in `.github/workflows/ci.yml`

## Repository Layout

The first tree below is historical. Use the canonical layout section that follows for the current repository structure.

```text
.
├── backend/
│   ├── app/                    # YouTube analysis API, models, persistence
│   ├── app_api/                # JWT serializer/view extensions
│   ├── core/                   # Django settings and URL configuration
│   ├── docs/                   # Project docs
│   ├── files/                  # Text resources used by preprocessing
│   ├── models/                 # Trained model artifacts
│   ├── research/               # Thesis / experiment code, not on main request path
│   ├── src/                    # Reusable preprocessing and sentiment engines
│   └── users/                  # Registration, profile, JWT alias/logout endpoints
├── frontend/
│   ├── src/
│   │   ├── Components/
│   │   ├── Views/
│   │   ├── context/
│   │   └── utils/
│   ├── package.json
│   └── vite.config.mjs
└── .github/workflows/ci.yml
```

Canonical layout:

```text
.
|- backend/
|  |- app/                  # Analysis API, models, persistence
|  |- app_api/              # JWT serializer/view extensions
|  |- core/                 # Django settings and URL configuration
|  |- data/                 # Datasets and split artifacts
|  |- docs/                 # Thesis and project documentation
|  |- figures/              # Generated plots and figures
|  |- models/               # Trained model artifacts
|  |- research/             # Thesis and experiment code
|  |- results/              # Generated reports and benchmark outputs
|  |- scripts/              # Data-prep utilities
|  |- src/                  # Reusable preprocessing and sentiment engines
|  |- tests/                # Backend test suites
|  `- users/                # Registration, profile, JWT alias/logout endpoints
|- frontend/
|  |- public/
|  |- src/
|  |- package.json
|  `- vite.config.mjs
|- .github/workflows/ci.yml
`- README.md
```

The repo root is intentionally thin. Backend Python manifests live in `backend/`, and frontend Node manifests live in `frontend/`.

## Quick Start

### Prerequisites

- Python 3.11–3.13 (the pinned `numpy==1.26.4` has no wheels for 3.14+, so
  `pip install -r requirements.txt` will fail to resolve on 3.14)
- Node.js 24+ and npm

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
python manage.py migrate
python manage.py runserver
```

The backend runs on `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

The frontend runs on `http://127.0.0.1:3000`.

Vite is configured to proxy `/api/*` to Django during local development. If you need a different API origin, set `VITE_API_URL` to a full `/api/` base, for example:

```bash
VITE_API_URL=http://127.0.0.1:8000/api/
```

## Environment

Backend env vars live in `backend/.env`.

| Variable | Required | Notes |
| --- | --- | --- |
| `DJANGO_ENV` | yes | Use `development` locally and `test` in CI. |
| `SECRET_KEY` | yes when `DEBUG=False` | Required in non-debug environments. |
| `DEBUG` | optional | Defaults are environment-sensitive. |
| `ALLOWED_HOSTS` | yes when `DEBUG=False` | Comma-separated list. |
| `CORS_ALLOWED_ORIGINS` | yes when `DEBUG=False` | Comma-separated allowlist. |
| `CSRF_TRUSTED_ORIGINS` | recommended | Usually mirrors `CORS_ALLOWED_ORIGINS`. |
| `YOUTUBE_API_KEY` | optional | Enables the official YouTube Data API path. |

The checked-in template is `backend/.env.example`.

## Main API Routes

All API routes are rooted at `/api/`. DRF defaults to authenticated access, with the auth bootstrap endpoints below explicitly public.

### Authentication

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/token/` | Primary JWT login endpoint. |
| `POST` | `/api/token/refresh/` | Rotate/refresh access tokens. |
| `POST` | `/api/user/register/` | Create a new user. |
| `POST` | `/api/user/logout/` | Blacklist a refresh token. |
| `GET` | `/api/user/me/<id>` | Return the authenticated user's profile and search history. |

### Analysis

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/youtube/analyze/` | Fetch comments, run sentiment analysis, persist the result. |
| `GET` | `/api/youtube/analysis/<video_id>/` | Fetch one saved analysis scoped to the current user. |
| `GET` | `/api/youtube/analyses/` | List the current user's saved analyses. |
| `GET` | `/api/youtube/health/` | Unauthenticated health check (DB reachability + default model artifacts present). Returns 503 if unhealthy. |

### Analysis Request Fields

Common request fields accepted by `/api/youtube/analyze/`:

- `video_url`
- `max_comments`
- `use_api`
- `sentiment_model`
- `ensemble_models`
- `ensemble_weights`
- `meta_learner_models`
- `confidence_threshold`
- `bootstrap_samples`
- `random_seed`
- `aspect_top_n`
- `aspect_min_freq`
- `fuzzy_models`
- `fuzzy_mf_type`
- `fuzzy_defuzz_method`
- `fuzzy_t_norm`
- `fuzzy_t_conorm`
- `fuzzy_alpha_cut`
- `fuzzy_resolution`
- `model_comparison`

Important constraints:

- `ensemble_weights` must be inline JSON. File paths are rejected.
- `meta_learner_path` is not accepted from API clients.
- Analysis lookups are scoped to the authenticated user.

## Frontend Routes

Primary routes defined in `frontend/src/App.jsx`:

- Public: `/`, `/signin`, `/register`
- Protected: `/search`, `/dashboard`, `/monitoring`, `/profile`, `/report/:name`
- Legacy pages still present in the router: `/tables`, `/editprofile`

## Testing

### Backend

```bash
cd backend
python manage.py check
python manage.py test app app_api users
```

### Frontend

```bash
cd frontend
npm test -- --run
```

Additional frontend testing notes live in `frontend/TESTING_GUIDE.md`.

## CI

GitHub Actions is configured in `.github/workflows/ci.yml` and runs:

- `python manage.py check`
- `python manage.py test app app_api users`
- `npm test -- --run`

on every `push` and `pull_request`.

## Security Notes

- JWT auth is the default backend authentication mode.
- Registration is public; most other API routes require authentication.
- Refresh tokens are blacklisted on logout.
- Production settings require an explicit `SECRET_KEY`, `ALLOWED_HOSTS`, and CORS allowlist.
- The backend no longer accepts client-supplied server-side model paths.

## Related Docs

- `backend/docs/ARCHITECTURE.md`
- `backend/docs/ROUTE_A_IMPLEMENTATION_ROADMAP.md`
- `backend/docs/THESIS_EXPERIMENT_GUIDE.md`
- `backend/docs/THESIS_RISKS_GAPS.md`
- `backend/README_THESIS.md`
- `backend/results/thesis_evaluation_report.json`
- `frontend/TESTING_GUIDE.md`

## License

MIT
