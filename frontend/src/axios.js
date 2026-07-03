import axios from "axios";
import { getStoredAccessToken } from "./utils/auth";

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

export default axiosInstance;
