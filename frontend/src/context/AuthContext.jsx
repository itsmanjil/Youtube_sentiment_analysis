import axiosInstance from "../axios";
import { createContext, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  clearStoredAuthToken,
  decodeAccessToken,
  getInitialAuthState,
  hasValidAccessToken,
  persistAuthToken,
  shouldRefreshAccessToken,
} from "../utils/auth";

const AuthContext = createContext();

export default AuthContext;

export const AuthProvider = ({ children }) => {
  const navigate = useNavigate();
  const initialAuthState = getInitialAuthState();

  let [authToken, setAuthToken] = useState(initialAuthState.authToken);
  let [user, setUser] = useState(initialAuthState.user);
  let [loading, setLoading] = useState(true);

  const [isError, SetIsError] = useState(false);

  const clearSession = () => {
    setAuthToken(null);
    setUser(null);
    clearStoredAuthToken();
  };

  const storeSession = (nextAuthToken) => {
    const decodedUser = decodeAccessToken(nextAuthToken?.access);
    if (!decodedUser) {
      clearSession();
      return false;
    }

    setAuthToken(nextAuthToken);
    setUser(decodedUser);
    persistAuthToken(nextAuthToken);
    return true;
  };

  let loginUser = async (email, password) => {
    try {
      // The refresh token is set as an httpOnly cookie by the backend
      // (core/auth_cookies.py) — it's never present in `data` here, and
      // axios.js sends withCredentials so the cookie round-trips correctly.
      let response = await axiosInstance.post("token/", {
        email: email,
        password: password,
      });

      let data = response.data;

      if (response.status === 200 && storeSession({ access: data?.access })) {
        SetIsError(false);
        navigate("/dashboard", { replace: true });
        return;
      }
    } catch {}

    SetIsError(true);
    clearSession();
    navigate("/signin", { replace: true });
  };

  let logoutUser = async (redirectTo = "/") => {
    try {
      // No body needed — the refresh cookie is attached automatically.
      await axiosInstance.post("user/logout/", {});
    } catch {}

    clearSession();
    setLoading(false);
    navigate(redirectTo, { replace: true });
  };

  let updateToken = async () => {
    try {
      // No body needed — the refresh cookie is attached automatically, and
      // the backend writes the rotated refresh token back to that cookie.
      let response = await axiosInstance.post("token/refresh/", {});
      let data = response.data;

      if (response.status === 200 && data?.access) {
        SetIsError(false);
        const stored = storeSession({ access: data.access });
        setLoading(false);
        return stored;
      }
    } catch {}

    SetIsError(true);
    clearSession();
    setLoading(false);
    navigate("/signin", { replace: true });
    return false;
  };

  const isAuthenticated = Boolean(user && hasValidAccessToken(authToken));

  let contextData = {
    isError: isError,
    isAuthenticated: isAuthenticated,
    loading: loading,
    user: user,
    authToken: authToken,
    loginUser: loginUser,
    logoutUser: logoutUser,
  };

  useEffect(() => {
    const initializeAuth = async () => {
      if (!authToken) {
        setLoading(false);
        return;
      }

      if (hasValidAccessToken(authToken)) {
        if (!user) {
          setUser(decodeAccessToken(authToken.access));
        }
        setLoading(false);
        return;
      }

      // Access is missing/expired but we have a prior authToken record —
      // there may still be a valid httpOnly refresh cookie, so attempt a
      // refresh rather than assuming the session is dead (updateToken()
      // itself clears the session and redirects to /signin on failure).
      await updateToken();
    };

    initializeAuth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!authToken) {
      return undefined;
    }

    let interval = setInterval(() => {
      if (shouldRefreshAccessToken(authToken)) {
        updateToken();
      }
    }, 1000 * 60);

    return () => clearInterval(interval);
  }, [authToken]);

  return (
    <AuthContext.Provider value={contextData}>{children}</AuthContext.Provider>
  );
};
