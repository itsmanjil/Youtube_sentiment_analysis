
from pathlib import Path
from datetime import timedelta
import os
import sys
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

from core.settings_utils import env_bool, resolve_database_settings, resolve_runtime_settings

_RUNNING_TESTS = "test" in sys.argv
DEFAULT_ENVIRONMENT = "test" if _RUNNING_TESTS else "production"
RUNTIME_SETTINGS = resolve_runtime_settings(
    os.environ,
    default_environment=DEFAULT_ENVIRONMENT,
    # `manage.py test` is unambiguous proof this is a test run — force it
    # regardless of a stray DJANGO_ENV in the environment/.env file (see
    # settings_utils._resolve_environment for why this matters).
    force_test=_RUNNING_TESTS,
)



# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = RUNTIME_SETTINGS["secret_key"]

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = RUNTIME_SETTINGS["debug"]

ALLOWED_HOSTS = RUNTIME_SETTINGS["allowed_hosts"]

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'app',
    'app_api',
    'rest_framework',
    'corsheaders',
    'rest_framework_simplejwt.token_blacklist',
    'users'
    
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'



DATABASES = resolve_database_settings(
    os.environ,
    debug=DEBUG,
    base_dir=BASE_DIR,
)




AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]



LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True



STATIC_URL = 'static/'

_IS_TEST_ENVIRONMENT = RUNTIME_SETTINGS["environment"] == "test"

# youtube/analyze/ can take far longer than any reasonable HTTP timeout for
# large max_comments/transformer models, so it normally runs on a background
# thread and returns a pollable job id (see app/models.py::AnalysisJob and
# app/views.py). Tests default to running it synchronously in-request so the
# large existing test suite (which asserts on the direct response body) does
# not need to poll a background job; override with ANALYSIS_RUN_SYNC=true/false
# to force either mode (e.g. for local debugging without a worker thread).
ANALYSIS_RUN_SYNC = env_bool(os.environ, "ANALYSIS_RUN_SYNC", default=_IS_TEST_ENVIRONMENT)

# An AnalysisJob's background thread can die without ever updating the job
# row (process restart/crash/OOM — the dev autoreloader alone does this on
# every code change during an in-flight analysis), leaving it stuck at
# status="running"/"pending" forever with no server-side record of what
# happened. Any job whose `updated_at` is older than this is treated as
# abandoned: `get_analysis_job_status` lazily marks it failed the next time
# anyone polls it, and `manage.py cleanup_stale_analysis_jobs` (run via an
# external scheduler) sweeps ones nobody ever polls again. Comfortably above
# the frontend's own ~6 minute poll timeout (Search.jsx: 180 * 2s).
STALE_ANALYSIS_JOB_TIMEOUT = timedelta(minutes=15)

REST_FRAMEWORK = {

    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated'
    ],

    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),

    # Unthrottled AllowAny endpoints (register/login) and the CPU-heavy
    # analyze endpoint (YouTube fetch + preprocessing + model inference +
    # bootstrap resampling) have no cost to hammering them otherwise.
    # `DEFAULT_THROTTLE_CLASSES` (anon/user) is skipped entirely in the test
    # environment since `manage.py test` reuses one IP/user across many
    # requests per run. `analyze` uses an explicit @throttle_classes on the
    # view (not gated by DEFAULT_THROTTLE_CLASSES), so its rate is always
    # defined here — just far looser in tests than in dev/production.
    'DEFAULT_THROTTLE_CLASSES': [] if _IS_TEST_ENVIRONMENT else [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '20/minute',
        'user': '60/minute',
        'analyze': '100000/day' if _IS_TEST_ENVIRONMENT else '10/hour',
    },
}

SIMPLE_JWT = {
    # Access token: kept short since it lives in localStorage
    # (XSS-exfiltratable) and the frontend already refreshes silently
    # (AuthContext polls every 60s and refreshes ~60s before expiry) — a
    # long-lived access token only adds risk without adding convenience.
    # Refresh token: no longer in localStorage at all — see
    # core/auth_cookies.py, which stores it in an httpOnly cookie instead,
    # so it isn't readable by JavaScript regardless of its lifetime.
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=1),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=90),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'UPDATE_LAST_LOGIN': False,

    'ALGORITHM': 'HS256',

    'VERIFYING_KEY': None,
    'AUDIENCE': None,
    'ISSUER': None,
    'JWK_URL': None,
    'LEEWAY': 0,

    'AUTH_HEADER_TYPES': ('Bearer',),
    'AUTH_HEADER_NAME': 'HTTP_AUTHORIZATION',
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'USER_AUTHENTICATION_RULE': 'rest_framework_simplejwt.authentication.default_user_authentication_rule',

    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
    'TOKEN_USER_CLASS': 'rest_framework_simplejwt.models.TokenUser',

    'JTI_CLAIM': 'jti',

    'SLIDING_TOKEN_REFRESH_EXP_CLAIM': 'refresh_exp',
    'SLIDING_TOKEN_LIFETIME': timedelta(minutes=5),
    'SLIDING_TOKEN_REFRESH_LIFETIME': timedelta(days=1),
}


DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CORS_ALLOW_ALL_ORIGINS = RUNTIME_SETTINGS["cors_allow_all_origins"]
CORS_ALLOWED_ORIGINS = RUNTIME_SETTINGS["cors_allowed_origins"]
CSRF_TRUSTED_ORIGINS = RUNTIME_SETTINGS["csrf_trusted_origins"]
# Required so the browser sends/accepts the httpOnly refresh-token cookie
# (core/auth_cookies.py) on cross-origin requests from the frontend dev
# server. Safe alongside CORS_ALLOW_ALL_ORIGINS=False + an explicit
# CORS_ALLOWED_ORIGINS list (enforced by settings_utils.resolve_runtime_settings
# in production) — browsers refuse credentialed requests against a wildcard
# origin anyway.
CORS_ALLOW_CREDENTIALS = True

SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"

# Custom user model
AUTH_USER_MODEL = "users.NewUser"
