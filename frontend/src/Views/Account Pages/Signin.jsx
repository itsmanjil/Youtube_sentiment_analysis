import { useContext, useState } from "react";
import { Link, Navigate } from "react-router-dom";
// Landing-theme styles (.ex-header, .ex-form-1, .offcanvas-collapse): needed
// for direct loads of /signin, which otherwise render an unstyled navbar.
import "../Pages/Homepage.css";

// import Navbar from "../../Components/Navbar";
import SigninForm from "./SigninForm";
import AuthContext from "../../context/AuthContext";
import usePageTitle from "../../utils/usePageTitle";

function Signin() {
  usePageTitle("Log In");
  const { isAuthenticated, loading } = useContext(AuthContext);
  const [isNavOpen, setIsNavOpen] = useState(false);

  if (!loading && isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <>
      {/* Navbar */}
      <nav
        id="navbarExample"
        className="navbar navbar-expand-lg fixed-top"
        aria-label="Main navigation"
      >
        <div className="container">
          {/* <!-- Image Logo --> */}
          <Link to="/" className="navbar-brand logo-image">
            <img
              src="../assets/img/logo2.png"
              alt="alternative"
              style={{ height: "40px", width: "40px" }}
            />
          </Link>
          <Link to="/" className="navbar-brand logo-text">
            YouTube Sentiment Analysis
          </Link>
          <button
            className="navbar-toggler p-0 border-0"
            type="button"
            id="navbarSideCollapse"
            aria-label="Toggle navigation"
            aria-expanded={isNavOpen}
            onClick={() => setIsNavOpen((open) => !open)}
          >
            <span className="navbar-toggler-icon"></span>
          </button>

          <div
            className={`navbar-collapse offcanvas-collapse${isNavOpen ? " open" : ""}`}
            id="navbarsExampleDefault"
            style={isNavOpen ? { visibility: "visible", transform: "translateX(-100%)" } : undefined}
            onClick={(e) => {
              if (e.target.closest("a")) {
                setIsNavOpen(false);
              }
            }}
          >
            <ul className="navbar-nav ms-auto navbar-nav-scroll">
              <li className="nav-item">
                <Link to="/" className="nav-link active" aria-current="page">
                  Home
                </Link>
              </li>
              
              

              {isAuthenticated && (
                <>
                  <li className="nav-item">
                    <Link to="/dashboard" className="nav-link" aria-current="page">
                      Dashboard
                    </Link>
                  </li>
                  <li className="nav-item">
                    <Link to="/profile" className="nav-link" aria-current="page">
                      Profile
                    </Link>
                  </li>
                </>
              )}
            </ul>
            {!isAuthenticated && (
              <span className="nav-item">
                <Link to="/signin" className="btn-outline-sm">
                  Log in
                </Link>
              </span>
            )}
          </div>
        </div>
      </nav>
      <header className="ex-header">
        <div className="container">
          <div className="row">
            <div className="col-xl-10 offset-xl-1">
              <h1 className="text-center">Log In</h1>
            </div>
          </div>
        </div>
      </header>

      <SigninForm />
    </>
  );
}

export default Signin;
