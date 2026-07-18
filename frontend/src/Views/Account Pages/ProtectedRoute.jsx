import { useContext } from "react";
import { Navigate, Outlet, useLocation } from "react-router-dom";
import AuthContext from "../../context/AuthContext";

export const ProtectedRoute = () => {
  const location = useLocation();
  const { isAuthenticated, loading } = useContext(AuthContext);

  if (loading) {
    return null;
  }

  return isAuthenticated ? (
    <Outlet />
  ) : (
    <Navigate to="/signin" replace state={{ from: location }} />
  );
};
