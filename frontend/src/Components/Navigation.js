import React from "react";
import { Link } from "react-router-dom";
import { HashLink } from "react-router-hash-link";
// Link for routing to diff pages & Hashlink for smooth transition to different parts of same page

function Navigation() {
  const token = localStorage.getItem("authToken");
  console.log("token", token);
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
          >
            <span className="navbar-toggler-icon"></span>
          </button>

          <div
            className="navbar-collapse offcanvas-collapse"
            id="navbarsExampleDefault"
          >
            <ul className="navbar-nav ms-auto navbar-nav-scroll">
              <li className="nav-item">
                <Link to="/" className="nav-link active" aria-current="page">
                  Home
                </Link>
              </li>
              

              {token !== null && (
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
            {token == null && (
              <span className="nav-item">
                <Link to="/signin" className="btn-outline-sm">
                  Log in
                </Link>
              </span>
            )}
          </div>
        </div>
      </nav>
      {/* End Navbar*/}
    </>
  );
}

export default Navigation;
