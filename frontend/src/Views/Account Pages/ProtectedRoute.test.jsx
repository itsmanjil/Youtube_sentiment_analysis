import { describe, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import { ProtectedRoute } from "./ProtectedRoute";
import AuthContext from "../../context/AuthContext";

const renderProtectedRoute = (contextValue) =>
  render(
    <MemoryRouter initialEntries={["/dashboard"]}>
      <AuthContext.Provider
        value={{
          isAuthenticated: false,
          loading: false,
          ...contextValue,
        }}
      >
        <Routes>
          <Route path="/signin" element={<div>signin-page</div>} />
          <Route element={<ProtectedRoute />}>
            <Route path="/dashboard" element={<div>dashboard-page</div>} />
          </Route>
        </Routes>
      </AuthContext.Provider>
    </MemoryRouter>
  );

describe("ProtectedRoute", () => {
  test("renders protected content for authenticated users", () => {
    renderProtectedRoute({ isAuthenticated: true });

    expect(screen.getByText("dashboard-page")).toBeInTheDocument();
  });

  test("redirects unauthenticated users to signin", () => {
    renderProtectedRoute({ isAuthenticated: false });

    expect(screen.getByText("signin-page")).toBeInTheDocument();
  });

  test("waits for auth initialization before redirecting", () => {
    renderProtectedRoute({ loading: true });

    expect(screen.queryByText("dashboard-page")).not.toBeInTheDocument();
    expect(screen.queryByText("signin-page")).not.toBeInTheDocument();
  });
});
