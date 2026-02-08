import axios from "axios";

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
  timeout: 1000 * 10,
  validateStatus: (status) => {
    // handling our own errors less than 500 status
    return status < 500;
  },
  headers: {
    "Content-Type": "application/json",
    accept: "application/json",
  },
});

const getAccessTokenFromStorage = () => {
  const raw = localStorage.getItem("authToken");
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw);
    return parsed?.access || null;
  } catch (err) {
    // Fallback: some code paths may store the raw access token string.
    return raw;
  }
};

// Ensure Authorization header stays in sync after login/refresh.
axiosInstance.interceptors.request.use((config) => {
  const accessToken = getAccessTokenFromStorage();
  if (accessToken) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${accessToken}`;
  } else if (config.headers && "Authorization" in config.headers) {
    delete config.headers.Authorization;
  }
  return config;
});

export default axiosInstance;
