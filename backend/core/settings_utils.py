from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured


LOCAL_ENVIRONMENTS = {"development", "dev", "local", "test"}
LOCAL_ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
LOCAL_CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
DEV_SECRET_KEY = "django-insecure-local-dev-only-change-me"


def env_bool(
    env: Mapping[str, str],
    name: str,
    default: bool = False,
) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def env_list(
    env: Mapping[str, str],
    name: str,
    default: Sequence[str] | None = None,
) -> list[str]:
    value = env.get(name)
    if value is None:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_environment(
    env: Mapping[str, str],
    default_environment: str = "production",
) -> str:
    value = env.get("DJANGO_ENV", env.get("ENVIRONMENT", default_environment))
    normalized = (value or default_environment).strip().lower()
    return normalized or default_environment


def resolve_runtime_settings(
    env: Mapping[str, str],
    default_environment: str = "production",
) -> dict[str, object]:
    environment = _resolve_environment(env, default_environment=default_environment)
    debug = env_bool(env, "DEBUG", default=environment in LOCAL_ENVIRONMENTS)

    secret_key = (env.get("SECRET_KEY") or "").strip()
    if not secret_key:
        if debug:
            secret_key = DEV_SECRET_KEY
        else:
            raise ImproperlyConfigured(
                "SECRET_KEY must be set when DEBUG is False."
            )

    allowed_hosts = env_list(
        env,
        "ALLOWED_HOSTS",
        default=LOCAL_ALLOWED_HOSTS if debug else [],
    )
    if not allowed_hosts:
        raise ImproperlyConfigured(
            "ALLOWED_HOSTS must be set when DEBUG is False."
        )

    cors_allow_all_origins = env_bool(
        env,
        "CORS_ALLOW_ALL_ORIGINS",
        default=env_bool(env, "CORS_ORIGIN_ALLOW_ALL", default=False),
    )
    cors_allowed_origins = env_list(
        env,
        "CORS_ALLOWED_ORIGINS",
        default=LOCAL_CORS_ALLOWED_ORIGINS if debug else [],
    )
    if not debug and cors_allow_all_origins:
        raise ImproperlyConfigured(
            "CORS_ALLOW_ALL_ORIGINS cannot be enabled when DEBUG is False."
        )
    if not debug and not cors_allow_all_origins and not cors_allowed_origins:
        raise ImproperlyConfigured(
            "CORS_ALLOWED_ORIGINS must be set when DEBUG is False."
        )

    csrf_trusted_origins = env_list(
        env,
        "CSRF_TRUSTED_ORIGINS",
        default=cors_allowed_origins,
    )

    return {
        "environment": environment,
        "debug": debug,
        "secret_key": secret_key,
        "allowed_hosts": allowed_hosts,
        "cors_allow_all_origins": cors_allow_all_origins,
        "cors_allowed_origins": cors_allowed_origins,
        "csrf_trusted_origins": csrf_trusted_origins,
    }


def resolve_database_settings(
    env: Mapping[str, str],
    *,
    debug: bool,
    base_dir: Path,
) -> dict[str, object]:
    """
    Resolve the Django DATABASES setting.

    SQLite (single-writer, file-locking) is fine for local development and
    tests, but is a concurrency/scaling hazard in production: concurrent
    analyze requests each issue many writes, and multi-worker deployments hit
    "database is locked" errors. `psycopg2-binary` is already a pinned
    dependency, so Postgres is wired up here via DB_ENGINE=postgres and the
    standard DB_NAME/DB_USER/DB_PASSWORD/DB_HOST/DB_PORT env vars. Production
    (DEBUG=False) must opt in explicitly rather than silently falling back to
    SQLite.
    """
    db_engine = (env.get("DB_ENGINE") or "").strip().lower()

    if db_engine in {"postgres", "postgresql"}:
        missing = [
            name
            for name in ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST")
            if not (env.get(name) or "").strip()
        ]
        if missing:
            raise ImproperlyConfigured(
                f"DB_ENGINE=postgres requires {missing} to be set."
            )
        return {
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": env["DB_NAME"].strip(),
                "USER": env["DB_USER"].strip(),
                "PASSWORD": env["DB_PASSWORD"],
                "HOST": env["DB_HOST"].strip(),
                "PORT": (env.get("DB_PORT") or "5432").strip(),
            }
        }

    if db_engine and db_engine != "sqlite":
        raise ImproperlyConfigured(
            f"Unsupported DB_ENGINE={db_engine!r}. Use 'postgres' or 'sqlite'."
        )

    if not debug:
        raise ImproperlyConfigured(
            "DB_ENGINE=postgres (with DB_NAME/DB_USER/DB_PASSWORD/DB_HOST) must "
            "be configured when DEBUG is False. SQLite is a single-writer, "
            "file-locking datastore and is not safe for a production deployment "
            "with concurrent requests."
        )

    return {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": base_dir / "db.sqlite3",
        }
    }
