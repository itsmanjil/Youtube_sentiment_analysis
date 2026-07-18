import { jwtDecode } from "jwt-decode";

export const AUTH_STORAGE_KEY = "authToken";
const CLOCK_SKEW_MS = 60 * 1000;

const looksLikeJwt = (value) =>
  typeof value === "string" && value.split(".").length === 3;

export const parseStoredAuthToken = (rawValue) => {
  if (!rawValue) {
    return null;
  }

  try {
    const parsed = JSON.parse(rawValue);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return looksLikeJwt(rawValue) ? { access: rawValue } : null;
  }
};

export const getStoredAuthToken = () =>
  parseStoredAuthToken(localStorage.getItem(AUTH_STORAGE_KEY));

export const persistAuthToken = (authToken) => {
  if (!authToken) {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    return;
  }

  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(authToken));
};

export const clearStoredAuthToken = () => {
  localStorage.removeItem(AUTH_STORAGE_KEY);
};

export const decodeAccessToken = (accessToken) => {
  if (!looksLikeJwt(accessToken)) {
    return null;
  }

  try {
    return jwtDecode(accessToken);
  } catch {
    return null;
  }
};

export const isAccessTokenExpired = (
  accessToken,
  skewMs = 0,
  nowMs = Date.now()
) => {
  const decoded = decodeAccessToken(accessToken);
  if (!decoded || typeof decoded.exp !== "number") {
    return true;
  }

  return decoded.exp * 1000 <= nowMs + skewMs;
};

export const hasValidAccessToken = (authToken, nowMs = Date.now()) =>
  Boolean(
    authToken?.access &&
      !isAccessTokenExpired(authToken.access, 0, nowMs)
  );

export const shouldRefreshAccessToken = (authToken, nowMs = Date.now()) => {
  // The refresh token lives in an httpOnly cookie now (see
  // core/auth_cookies.py on the backend) — JS can't see whether one exists,
  // so "should we attempt a refresh" is decided purely by whether our
  // locally-stored access token is missing/near-expiry, for any session we
  // believe is active (authToken record exists at all).
  if (!authToken) {
    return false;
  }

  return isAccessTokenExpired(authToken.access, CLOCK_SKEW_MS, nowMs);
};

export const getStoredAccessToken = () => {
  const authToken = getStoredAuthToken();
  return hasValidAccessToken(authToken) ? authToken.access : null;
};

export const getInitialAuthState = () => {
  const authToken = getStoredAuthToken();
  if (!authToken) {
    return { authToken: null, user: null };
  }

  const user = decodeAccessToken(authToken.access);
  if (user && !isAccessTokenExpired(authToken.access)) {
    return { authToken, user };
  }

  // Access is missing/expired, but a valid refresh cookie may still exist
  // (it's httpOnly, so JS can't check) — keep the record so AuthContext's
  // init effect attempts a refresh instead of assuming the session is dead.
  return { authToken, user: null };
};
