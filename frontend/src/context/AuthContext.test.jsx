import { useContext } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import AuthContext, { AuthProvider } from "./AuthContext";
import axiosInstance from "../axios";

vi.mock("../axios", () => ({
  default: {
    post: vi.fn(),
  },
}));

const mockNavigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual("react-router-dom");
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

const encodeSegment = (value) =>
  btoa(JSON.stringify(value))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");

const makeToken = (payload) =>
  `${encodeSegment({ alg: "HS256", typ: "JWT" })}.${encodeSegment(payload)}.signature`;

const ContextProbe = () => {
  const { isAuthenticated, loading, user } = useContext(AuthContext);

  if (loading) {
    return <div>loading</div>;
  }

  if (!isAuthenticated) {
    return <div>signed-out</div>;
  }

  return <div>{user?.user_name}</div>;
};

const renderWithProvider = () =>
  render(
    <MemoryRouter>
      <AuthProvider>
        <ContextProbe />
      </AuthProvider>
    </MemoryRouter>
  );

describe("AuthProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("silently restores a session on mount via the refresh cookie", async () => {
    // Nothing is read from storage on mount (the access token lives in
    // memory only) — the provider always attempts one silent refresh via
    // the httpOnly cookie (axios.js sends withCredentials) to find out
    // whether a session actually exists.
    const refreshedAccess = makeToken({
      exp: Math.floor(Date.now() / 1000) + 600,
      user_name: "Fresh User",
    });
    axiosInstance.post.mockResolvedValueOnce({
      status: 200,
      data: { access: refreshedAccess },
    });

    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByText("Fresh User")).toBeInTheDocument();
    });

    expect(axiosInstance.post).toHaveBeenCalledWith("token/refresh/", {});
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  test("does not redirect an anonymous visitor when there is no session to restore", async () => {
    // A fresh/anonymous visit looks identical to an expired session at
    // mount time (no local record either way) -- the silent refresh
    // failing here must not force a redirect to /signin, or every
    // anonymous page load would get bounced off public pages.
    axiosInstance.post.mockRejectedValueOnce(new Error("no refresh cookie"));

    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByText("signed-out")).toBeInTheDocument();
    });

    expect(mockNavigate).not.toHaveBeenCalled();
  });

  test("redirects to signin when an active session's refresh later fails", async () => {
    // Once a session is genuinely established, a later refresh failure
    // (e.g. the 90-day refresh cookie finally expiring) should still
    // redirect -- that behavior lives in the periodic refresh path now
    // instead of the mount-time path.
    vi.useFakeTimers();
    try {
      const nearExpiryAccess = makeToken({
        exp: Math.floor(Date.now() / 1000) + 30,
        user_name: "Soon Expiring",
      });
      axiosInstance.post.mockResolvedValueOnce({
        status: 200,
        data: { access: nearExpiryAccess },
      });

      renderWithProvider();

      await vi.waitFor(() => {
        expect(screen.getByText("Soon Expiring")).toBeInTheDocument();
      });

      axiosInstance.post.mockRejectedValueOnce(new Error("refresh failed"));

      await vi.advanceTimersByTimeAsync(60 * 1000);

      await vi.waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith("/signin", {
          replace: true,
        });
        expect(screen.getByText("signed-out")).toBeInTheDocument();
      });
    } finally {
      vi.useRealTimers();
    }
  });
});
