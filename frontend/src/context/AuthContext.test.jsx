import React, { useContext } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
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
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  test("refreshes an expired stored session on startup", async () => {
    const expiredAccess = makeToken({
      exp: Math.floor(Date.now() / 1000) - 60,
      user_name: "Old User",
    });
    const refreshedAccess = makeToken({
      exp: Math.floor(Date.now() / 1000) + 600,
      user_name: "Fresh User",
    });
    localStorage.setItem(
      "authToken",
      JSON.stringify({ access: expiredAccess, refresh: "refresh-1" })
    );
    axiosInstance.post.mockResolvedValueOnce({
      status: 200,
      data: { access: refreshedAccess, refresh: "refresh-2" },
    });

    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByText("Fresh User")).toBeInTheDocument();
    });

    expect(axiosInstance.post).toHaveBeenCalledWith("token/refresh/", {
      refresh: "refresh-1",
    });
    expect(JSON.parse(localStorage.getItem("authToken"))).toEqual({
      access: refreshedAccess,
      refresh: "refresh-2",
    });
  });

  test("clears session and redirects to signin when refresh fails", async () => {
    const expiredAccess = makeToken({
      exp: Math.floor(Date.now() / 1000) - 60,
      user_name: "Old User",
    });
    localStorage.setItem(
      "authToken",
      JSON.stringify({ access: expiredAccess, refresh: "refresh-1" })
    );
    axiosInstance.post.mockRejectedValueOnce(new Error("refresh failed"));

    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByText("signed-out")).toBeInTheDocument();
    });

    expect(localStorage.getItem("authToken")).toBeNull();
    expect(mockNavigate).toHaveBeenCalledWith("/signin", { replace: true });
  });
});
