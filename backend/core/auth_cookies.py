"""
Shared helpers for storing the JWT refresh token in an httpOnly cookie
instead of the response body.

Why: the access token still goes in the JSON body (short-lived, held only in
memory/localStorage by the frontend), but the refresh token is long-lived
(SIMPLE_JWT["REFRESH_TOKEN_LIFETIME"], 90 days) and was previously returned in
the body for the frontend to store in localStorage — readable by any script
on the page, so an XSS bug anywhere in the frontend meant a 90-day account
takeover. An httpOnly cookie is invisible to JavaScript entirely, closing
that window; SameSite="Lax" additionally means the cookie is not attached to
cross-site POST requests (only top-level GET navigations), which is a
sufficient CSRF mitigation for a login/refresh/logout flow that never runs on
GET. This does mean the frontend and backend must be same-site in production
(e.g. api.example.com + app.example.com, or same origin behind one proxy) —
a cross-site deployment would need SameSite="None" + a separate CSRF
mitigation instead.
"""

from __future__ import annotations

from django.conf import settings

REFRESH_COOKIE_NAME = "refresh_token"
# Scoped to /api/ (not the whole site) since it's only needed by
# /api/token/refresh/ and /api/user/logout/, both of which read it from
# request.COOKIES rather than the request body.
REFRESH_COOKIE_PATH = "/api/"


def set_refresh_cookie(response, refresh_token) -> None:
    lifetime = settings.SIMPLE_JWT["REFRESH_TOKEN_LIFETIME"]
    response.set_cookie(
        REFRESH_COOKIE_NAME,
        str(refresh_token),
        max_age=int(lifetime.total_seconds()),
        httponly=True,
        secure=not settings.DEBUG,
        samesite="Lax",
        path=REFRESH_COOKIE_PATH,
    )


def clear_refresh_cookie(response) -> None:
    response.delete_cookie(REFRESH_COOKIE_NAME, path=REFRESH_COOKIE_PATH)
