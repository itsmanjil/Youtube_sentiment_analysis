import { jwtDecode } from "jwt-decode";

const CLOCK_SKEW_MS = 60 * 1000;

const looksLikeJwt = (value) =>
  typeof value === "string" && value.split(".").length === 3;

// The access token lives in memory only (this module-level variable) rather
// than localStorage: localStorage is readable by any JS running on the page,
// so an XSS bug anywhere would hand an attacker a live bearer token that
// stays valid (and exfiltratable by re-reading storage) until it expires.
// Keeping it in memory means it disappears the moment the tab/process ends,
// and a full page reload always starts from zero. The refresh token never
// reaches the frontend at all — it's an httpOnly cookie set by the backend
// (see core/auth_cookies.py) — so losing the in-memory access token on
// reload is recoverable: AuthContext's mount effect silently exchanges that
// cookie for a fresh access token via POST token/refresh/.
let inMemoryAuthToken = null;

export const getStoredAuthToken = () => inMemoryAuthToken;

export const persistAuthToken = (authToken) => {
  inMemoryAuthToken = authToken || null;
};

export const clearStoredAuthToken = () => {
  inMemoryAuthToken = null;
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

// Lets axios.js trigger a real refresh (POST token/refresh/ + update the
// in-memory token + React state) without importing AuthContext directly —
// axios.js has no React tree to sit inside, and AuthContext already owns
// the axios instance, so the dependency runs the other way: AuthContext
// registers its refresh function here once, on mount.
let refreshHandler = null;

export const registerRefreshHandler = (handler) => {
  refreshHandler = handler;
};

export const requestSilentRefresh = () => {
  if (!refreshHandler) {
    return Promise.resolve(false);
  }
  return refreshHandler();
};
