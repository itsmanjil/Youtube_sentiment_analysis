import { afterEach, describe, expect, test } from "vitest";
import {
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
    const authToken = { access: "a" };

    expect(parseStoredAuthToken(JSON.stringify(authToken))).toEqual(authToken);
  });

  test("keeps an expired access token so a cookie-based refresh can be attempted", () => {
    // The refresh token lives in an httpOnly cookie (invisible to JS), so
    // there's no client-visible "refresh" field to check anymore — an
    // expired/missing access token is always kept around long enough for
    // AuthContext to attempt a refresh; only a failed refresh clears it.
    const expiredAccess = makeToken({
      exp: Math.floor(Date.now() / 1000) - 60,
      user_name: "Expired",
    });

    persistAuthToken({ access: expiredAccess });

    expect(getInitialAuthState()).toEqual({
      authToken: { access: expiredAccess },
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

    expect(shouldRefreshAccessToken({ access: nearExpiryAccess })).toBe(true);
    expect(shouldRefreshAccessToken({ access: healthyAccess })).toBe(false);
    expect(shouldRefreshAccessToken(null)).toBe(false);
  });
});
