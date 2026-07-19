import axios from "axios";
import { getStoredAccessToken, requestSilentRefresh } from "./utils/auth";

// Prefer Vite dev proxy (`/api`) but allow override for deployments.
// Example: set `VITE_API_URL=http://127.0.0.1:8000/api/`
const normalizeBaseUrl = (raw) => {
  if (!raw) return "";
  const trimmed = String(raw).trim();
  if (!trimmed) return "";
  return trimmed.endsWith("/") ? trimmed : `${trimmed}/`;
};

const baseURL = normalizeBaseUrl(import.meta.env?.VITE_API_URL) || "/api/";

const axiosInstance = axios.create({
  baseURL: baseURL,
  timeout: 1000 * 120,
  // Required so the browser sends/accepts the httpOnly refresh-token cookie
  // (backend: core/auth_cookies.py) on cross-origin requests to the Django
  // dev server; the backend pairs this with CORS_ALLOW_CREDENTIALS=True and
  // an explicit CORS_ALLOWED_ORIGINS list (never a wildcard).
  withCredentials: true,
  validateStatus: (status) => {
    // handling our own errors less than 500 status
    return status < 500;
  },
  headers: {
    "Content-Type": "application/json",
    accept: "application/json",
  },
});

// Ensure Authorization header stays in sync after login/refresh.
axiosInstance.interceptors.request.use((config) => {
  const accessToken = getStoredAccessToken();
  if (accessToken) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${accessToken}`;
  } else if (config.headers && "Authorization" in config.headers) {
    delete config.headers.Authorization;
  }
  return config;
});

// `validateStatus` above treats every sub-500 status as a resolved
// response, not a rejected promise -- so a 401 lands in this *success*
// handler, not an error handler. The access token lives in memory only
// (utils/auth.js) and is proactively refreshed ~60s before it expires
// (AuthContext's poll), but that poll can miss a tick in a backgrounded
// tab (browsers throttle setInterval there), so a request can still land
// with a stale token. Rather than surface that as a failed request, try
// one silent refresh and replay the request with the new token.
axiosInstance.interceptors.response.use(async (response) => {
  const isAuthEndpoint = /^token\/?(refresh\/?)?$/.test(
    response.config?.url ?? ""
  );
  if (response.status !== 401 || response.config?._retriedAfterRefresh || isAuthEndpoint) {
    return response;
  }

  response.config._retriedAfterRefresh = true;
  const refreshed = await requestSilentRefresh();
  if (!refreshed) {
    return response;
  }

  // Re-dispatching the same config runs it back through the request
  // interceptor above, which reads getStoredAccessToken() fresh -- by now
  // pointing at the token requestSilentRefresh() just stored -- so the
  // retried request picks up the new Authorization header on its own.
  return axiosInstance(response.config);
});

export default axiosInstance;
