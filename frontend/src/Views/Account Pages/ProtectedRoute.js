import React from "react";
import { Navigate, Outlet } from "react-router-dom";

export const ProtectedRoute = () => {
  const loggedIn = localStorage.getItem("authToken") ? true : false;
  // console.log("loggedIn", loggedIn);
  return loggedIn ? <Outlet /> : <Navigate to="/signin" />;
};
