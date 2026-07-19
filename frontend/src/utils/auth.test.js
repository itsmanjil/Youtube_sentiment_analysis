import { afterEach, describe, expect, test } from "vitest";
import {
  clearStoredAuthToken,
  getStoredAuthToken,
  hasValidAccessToken,
  isAccessTokenExpired,
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
    clearStoredAuthToken();
  });

  test("holds the access token in memory, not localStorage", () => {
    const authToken = { access: "a" };

    persistAuthToken(authToken);

    expect(getStoredAuthToken()).toEqual(authToken);
    expect(localStorage.getItem("authToken")).toBeNull();

    clearStoredAuthToken();

    expect(getStoredAuthToken()).toBeNull();
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
