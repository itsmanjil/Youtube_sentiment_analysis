import axiosInstance from "../axios";
import { createContext, useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  clearStoredAuthToken,
  decodeAccessToken,
  hasValidAccessToken,
  persistAuthToken,
  registerRefreshHandler,
  shouldRefreshAccessToken,
} from "../utils/auth";

const AuthContext = createContext();

export default AuthContext;

export const AuthProvider = ({ children }) => {
  const navigate = useNavigate();

  // The access token lives in memory only (see utils/auth.js), so there is
  // never a stored value to read on mount — every fresh load starts signed
  // out here and, if a session actually exists, gets restored a moment
  // later via the httpOnly refresh cookie (see the init effect below).
  let [authToken, setAuthToken] = useState(null);
  let [user, setUser] = useState(null);
  let [loading, setLoading] = useState(true);

  const [isError, SetIsError] = useState(false);

  // Guards against two refresh attempts racing each other -- e.g. the
  // mount-time silent refresh and an axios 401 retry (see axios.js) firing
  // around the same moment. SimpleJWT rotates + blacklists the refresh
  // cookie on every use, so two concurrent refresh calls would make the
  // second one fail against an already-rotated cookie. A plain ref (not
  // state) is fine here: AuthProvider is mounted once for the app's
  // lifetime, and this doesn't need to trigger a re-render.
  const refreshInFlightRef = useRef(null);

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

  // `redirectOnFailure` is false only for the silent mount-time attempt
  // below: at that point we don't yet know if this is an expired session
  // or just an anonymous visitor on a public page, and forcing everyone
  // through /signin on every page load would be wrong for the latter.
  // ProtectedRoute already redirects unauthenticated visitors away from
  // protected pages once `loading` resolves, so no explicit navigation is
  // needed here for that case. The periodic/explicit refresh calls (a
  // session that was live and then died) keep redirecting as before.
  let updateToken = ({ redirectOnFailure = true } = {}) => {
    if (refreshInFlightRef.current) {
      return refreshInFlightRef.current;
    }

    const attempt = (async () => {
      try {
        // No body needed — the refresh cookie is attached automatically,
        // and the backend writes the rotated refresh token back to it.
        let response = await axiosInstance.post("token/refresh/", {});
        let data = response.data;

        if (response.status === 200 && data?.access) {
          SetIsError(false);
          const stored = storeSession({ access: data.access });
          setLoading(false);
          return stored;
        }
      } catch {}

      clearSession();
      setLoading(false);
      if (redirectOnFailure) {
        SetIsError(true);
        navigate("/signin", { replace: true });
      }
      return false;
    })();

    refreshInFlightRef.current = attempt;
    attempt.finally(() => {
      refreshInFlightRef.current = null;
    });
    return attempt;
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
    // Nothing is persisted locally, so the only way to know whether this
    // browser has a live session is to ask: attempt one silent refresh via
    // the httpOnly cookie. `redirectOnFailure: false` keeps a failed
    // attempt from bouncing an anonymous visitor to /signin.
    updateToken({ redirectOnFailure: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // Lets axios.js's response interceptor (see axios.js) trigger a real
    // refresh-and-retry when a request 401s between poll ticks, instead of
    // the request just failing outright. Re-registers every render so the
    // handler always closes over the latest state setters/navigate.
    registerRefreshHandler(() => updateToken());
  });

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
    // updateToken is deliberately not in the deps array: it captures no
    // render-scoped state that goes stale (no authToken read inside it --
    // refreshInFlightRef is a ref, always current; the refresh call itself
    // is cookie-based). Adding it would tear down and rebuild this interval
    // on every render instead of only when authToken changes, which can
    // starve the 60s check entirely if renders happen more often than that.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authToken]);

  return (
    <AuthContext.Provider value={contextData}>{children}</AuthContext.Provider>
  );
};
