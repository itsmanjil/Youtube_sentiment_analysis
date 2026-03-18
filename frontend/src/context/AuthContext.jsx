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
      let response = await axiosInstance.post("token/", {
        email: email,
        password: password,
      });

      let data = response.data;

      if (response.status === 200 && storeSession(data)) {
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
    const refreshToken = authToken?.refresh;
    if (refreshToken) {
      try {
        await axiosInstance.post("user/logout/", {
          refresh: refreshToken,
        });
      } catch {}
    }

    clearSession();
    setLoading(false);
    navigate(redirectTo, { replace: true });
  };

  let updateToken = async () => {
    if (!authToken?.refresh) {
      if (!hasValidAccessToken(authToken)) {
        clearSession();
      }
      setLoading(false);
      return false;
    }

    try {
      let response = await axiosInstance.post("token/refresh/", {
        refresh: authToken.refresh,
      });
      let data = response.data;

      if (response.status === 200 && data?.access) {
        const nextAuthToken = {
          ...authToken,
          ...data,
          refresh: data.refresh || authToken.refresh,
        };
        SetIsError(false);
        const stored = storeSession(nextAuthToken);
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

      if (authToken.refresh) {
        await updateToken();
        return;
      }

      clearSession();
      setLoading(false);
    };

    initializeAuth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!authToken?.refresh) {
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
