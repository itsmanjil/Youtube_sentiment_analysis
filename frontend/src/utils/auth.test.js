import { afterEach, describe, expect, test } from "vitest";
import {
  AUTH_STORAGE_KEY,
  getInitialAuthState,
  hasValidAccessToken,
  isAccessTokenExpired,
  parseStoredAuthToken,
  persistAuthToken,
  shouldRefreshAccessToken,
} from "./auth";

const encodeSegment = (value) =>
  btoa(JSON.stringify(value))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");

const makeToken = (payload) =>
  `${encodeSegment({ alg: "HS256", typ: "JWT" })}.${encodeSegment(payload)}.signature`;

describe("auth utils", () => {
  afterEach(() => {
    localStorage.clear();
  });

  test("parses JSON tokens from storage", () => {
    const authToken = { access: "a", refresh: "r" };

    expect(parseStoredAuthToken(JSON.stringify(authToken))).toEqual(authToken);
  });

  test("treats expired access token without refresh as signed out", () => {
    const expiredAccess = makeToken({
      exp: Math.floor(Date.now() / 1000) - 60,
      user_name: "Expired",
    });

    persistAuthToken({ access: expiredAccess });

    expect(getInitialAuthState()).toEqual({ authToken: null, user: null });
    expect(localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  test("keeps expired access token if refresh token exists", () => {
    const expiredAccess = makeToken({
      exp: Math.floor(Date.now() / 1000) - 60,
      user_name: "Refreshable",
    });

    persistAuthToken({ access: expiredAccess, refresh: "refresh-token" });

    expect(getInitialAuthState()).toEqual({
      authToken: { access: expiredAccess, refresh: "refresh-token" },
      user: null,
    });
  });

  test("detects valid access tokens", () => {
    const access = makeToken({
      exp: Math.floor(Date.now() / 1000) + 300,
      user_name: "Valid User",
    });

    expect(hasValidAccessToken({ access })).toBe(true);
    expect(isAccessTokenExpired(access)).toBe(false);
  });

  test("requests refresh when access token is close to expiry", () => {
    const nearExpiryAccess = makeToken({
      exp: Math.floor(Date.now() / 1000) + 30,
      user_name: "Soon Expiring",
    });
    const healthyAccess = makeToken({
      exp: Math.floor(Date.now() / 1000) + 600,
      user_name: "Healthy",
    });

    expect(
      shouldRefreshAccessToken({
        access: nearExpiryAccess,
        refresh: "refresh-token",
      })
    ).toBe(true);
    expect(
      shouldRefreshAccessToken({
        access: healthyAccess,
        refresh: "refresh-token",
      })
    ).toBe(false);
  });
});
