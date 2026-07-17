import { lazy, Suspense } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import { ProtectedRoute } from "./Views/Account Pages/ProtectedRoute";
import { AuthProvider } from "./context/AuthContext";

const Dashboard = lazy(() => import("./Views/Pages/Dashboard"));
const Signin = lazy(() => import("./Views/Account Pages/Signin"));
const Register = lazy(() => import("./Views/Account Pages/Register"));
const Profile = lazy(() => import("./Views/Account Pages/Profile"));
const Homepage = lazy(() => import("./Views/Pages/Homepage"));
const Search = lazy(() => import("./Views/Pages/Search"));
const Report = lazy(() => import("./Views/Pages/Report"));
const Monitoring = lazy(() => import("./Views/Pages/Monitoring"));

function RouteFallback() {
  return (
    <div className="min-vh-100 d-flex align-items-center justify-content-center">
      <span className="text-secondary">Loading...</span>
    </div>
  );
}

function NotFound() {
  return (
    <div className="min-vh-100 d-flex flex-column align-items-center justify-content-center text-center">
      <h1>404</h1>
      <p className="text-secondary">This page doesn't exist.</p>
      <Link to="/">Go home</Link>
    </div>
  );
}

function App() {
  return (
    <>
      <Router>
        <AuthProvider>
          <Suspense fallback={<RouteFallback />}>
            <Routes>
              <Route exact path="/" element={<Homepage />}></Route>
              <Route path="/signin" element={<Signin />}></Route>
              <Route path="/register" element={<Register />}></Route>
              <Route element={<ProtectedRoute />}>
                <Route path="/search" element={<Search />}></Route>
                <Route path="/dashboard" element={<Dashboard />}></Route>
                <Route path="/monitoring" element={<Monitoring />}></Route>
                <Route path="/profile" element={<Profile />}></Route>
                <Route exact path="/report/:videoId" element={<Report />}></Route>
              </Route>
              <Route path="*" element={<NotFound />}></Route>
            </Routes>
          </Suspense>
        </AuthProvider>
      </Router>
    </>
  );
}

export default App;
