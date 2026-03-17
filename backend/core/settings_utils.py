from __future__ import annotations

from collections.abc import Mapping, Sequence

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
