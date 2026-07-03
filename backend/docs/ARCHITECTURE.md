# System Architecture

This document describes the runtime architecture that is actually present in the repository today. It focuses on the web application request path first, then the research layer that lives alongside it.

## Runtime Overview

```text
React/Vite frontend
    |
    |  /api/*
    v
Django REST Framework API
    |
    +--> JWT auth / profile endpoints
    |
    +--> YouTube analysis endpoints
            |
            +--> comment fetch (YouTube API or scraper)
            +--> preprocessing / filtering
            +--> sentiment engine selection
            +--> analytics aggregation
            +--> persistence to SQLite
```

## Main Components

### Frontend

- `frontend/src/App.jsx` defines the main routes.
- `frontend/src/context/AuthContext.jsx` owns login, logout, refresh, and auth bootstrap.
- `frontend/src/axios.js` uses `/api/` by default and attaches non-expired bearer tokens.
- `frontend/src/Views/Pages/Search.jsx` collects analysis parameters and calls `/api/youtube/analyze/`.
- `frontend/src/Views/Pages/Dashboard.jsx`, `Monitoring.jsx`, and `Report.jsx` render saved or freshly returned analysis data.

### Backend

- `backend/core/urls.py` wires the API surface together.
- `backend/app/` owns the main analysis endpoints and database models.
- `backend/users/` owns registration, logout, and the user profile endpoint. (JWT login itself is issued by `app_api` at `/api/token/`.)
- `backend/app_api/` extends SimpleJWT so issued tokens include `user_name` and `is_registered`.
- `backend/src/` contains reusable preprocessing helpers and sentiment engines.

### Research Layer

- `backend/research/` contains experiment, benchmark, visualization, explainability, and computational intelligence code.
- This codebase is important for thesis work, but it is not on the critical path for a normal API request.

## Repository Shape

```text
backend/
├── app/                    # Analysis endpoints, ORM models, persistence
├── app_api/                # JWT serializer/view customization
├── core/                   # Settings, settings helpers, root URLs
├── docs/                   # Architecture and gap docs
├── files/                  # Language resources / preprocessing assets
├── models/                 # Trained model artifacts
├── research/               # Thesis and experiment code
├── src/                    # Shared preprocessing + sentiment engine code
└── users/                  # User registration/profile/auth alias endpoints

frontend/
├── src/Components/
├── src/Views/
├── src/context/
├── src/utils/
├── package.json
└── vite.config.mjs
```

## Request Flows

### 1. Authentication Flow

```text
Signin form
    -> POST /api/token/
    -> access + refresh JWT pair
    -> AuthContext stores session
    -> axios request interceptor sends Bearer token
```

Logout posts the refresh token to `/api/user/logout/`, which blacklists it server-side.

### 2. Analysis Submission Flow

```text
Search page
    -> POST /api/youtube/analyze/
    -> normalize request options
    -> fetch video metadata + comments
    -> preprocess/filter comments
    -> select sentiment engine
    -> run inference
    -> compute analytics
    -> save analysis + comments
    -> return response to frontend
```

The frontend can request classical models, ensemble variants, and research-oriented options such as fuzzy ensemble configuration. The backend now only accepts inline structured configuration for user-controlled options like `ensemble_weights`.

### 3. Analysis Retrieval Flow

```text
Monitoring / Dashboard
    -> GET /api/youtube/analyses/
    -> list current user's saved analyses

Monitoring detail action
    -> GET /api/youtube/analysis/<video_id>/
    -> fetch one saved analysis for the current user
```

Saved analysis retrieval is scoped to the authenticated user, so the same external YouTube video ID does not expose another user's latest result.

## Backend Analysis Pipeline

The main API path in `backend/app/views.py` follows this sequence:

1. Validate request parameters.
2. Normalize model aliases and optional experiment settings.
3. Fetch video metadata and comments with either the YouTube API or scraper path.
4. Preprocess comments:
   - spam filtering
   - language filtering
   - text normalization
   - short-comment removal
5. Select a sentiment engine from `backend/src/sentiment/`.
6. Run batch inference and collect per-comment sentiment output.
7. Aggregate analytics:
   - sentiment totals
   - weighted/top comment summaries
   - word-frequency output
   - aspect sentiment
   - confidence statistics and intervals
   - timeline data
8. Persist `YouTubeVideo`, `YouTubeComment`, and `YouTubeAnalysis` records.

### Runtime Artifact Pinning

Live CI behavior is now resolved from a pinned manifest instead of mutable root
research outputs:

- manifest: `backend/results/runtime/route_a_live_v1/manifest.json`
- calibrated temperatures: `temperature_scaling.json`
- PSO weights: `pso_ensemble_weights.json`
- NSGA-II weights: `multi_objective_ensemble.json`
- neuro-fuzzy gate: `neuro_fuzzy_gate.json`

This keeps thesis-facing runtime inference tied to a named artifact version. The
selected version is exposed in `analysis_meta["runtime_artifacts"]` for each
saved analysis.

## Data Model Responsibilities

At a high level:

- `NewUser` stores account data.
- `YouTubeVideo` stores normalized video metadata.
- `YouTubeComment` stores fetched/processed comments.
- `YouTubeAnalysis` stores a user's saved analysis snapshot for a video and model configuration.

The frontend relies on both direct analysis responses and previously saved `YouTubeAnalysis` records for dashboard and monitoring views.

## Security Boundaries

Recent hardening changed the architecture in a few important ways:

- DRF defaults to authenticated access.
- Registration is explicitly public; profile access is user-scoped.
- API clients cannot submit server filesystem paths for model artifacts.
- Logout uses refresh-token blacklisting instead of Django session logout.
- Production settings are environment-driven and reject insecure defaults.

## Settings and Environment

`backend/core/settings.py` delegates runtime policy resolution to `backend/core/settings_utils.py`.

Key behaviors:

- local/test environments can use safe development defaults
- production requires explicit `SECRET_KEY`
- production requires `ALLOWED_HOSTS`
- production requires explicit CORS allowlists
- secure cookie/browser headers are enabled when `DEBUG` is false

## Testing and CI

The project currently validates the main web app with:

- backend: `python manage.py check`
- backend: `python manage.py test app app_api users`
- frontend: `npm test -- --run`

Those commands are enforced in `.github/workflows/ci.yml`.

## What This Document Does Not Cover

- Detailed thesis experiment design in `backend/research/`
- Training procedures for every model artifact under `backend/models/`
- Deployment-specific infrastructure such as containers, reverse proxies, or cloud hosting

Those concerns belong in separate operational or research docs.
