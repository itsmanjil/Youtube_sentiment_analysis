import React, {useState} from "react";
import { Link } from "react-router-dom";
import { HashLink } from "react-router-hash-link";
// Link for routing to diff pages & Hashlink for smooth transition to different parts of same page


export default function Navbar() {
  const [isActive,setActive] = useState("home")
  
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
                <Link to="/" className={isActive==='home' ? 'nav-link active' : 'nav-link' } aria-current="page"
                onClick={() => setActive("home")}
                value={` ${isActive === "home" ? 'btn__nav-bar-btn active-link' : 'btn__nav-bar-btn'}`}
                >
                  Home
                </Link>
              </li>
              <li className="nav-item">
                <HashLink smooth to="#features" className={isActive==='features' ? 'nav-link active' : 'nav-link' }
                onClick={() => setActive("features")}
                value={` ${isActive === "features" ? 'btn__nav-bar-btn active-link' : 'btn__nav-bar-btn'}`}>
                  Features
                </HashLink>
              </li>
              <li className="nav-item">
                <HashLink smooth to="#details"  href="#details" className={isActive==='details' ? 'nav-link active' : 'nav-link' }
                onClick={() => setActive("details")}
                value={` ${isActive === "features" ? 'btn__nav-bar-btn active-link' : 'btn__nav-bar-btn'}`}>
                  Details
                </HashLink>
              </li>
              <li className="nav-item">
                <HashLink
                  smooth
                  to="#contact"
                  href="#details"
                  className={isActive==='contact' ? 'nav-link active' : 'nav-link' }
                  onClick={() => setActive("contact")}
                  value={` ${isActive === "contact" ? 'btn__nav-bar-btn active-link' : 'btn__nav-bar-btn'}`}
                >
                  Contact
                </HashLink>
              </li>

              {token !== null && (
                <>
                  <li className="nav-item">
                    <Link to="/dashboard" className="nav-link" aria-current="page">
                      Dashboard
                    </Link>
                  </li>
                  <li className="nav-item">
                    <Link to="/monitoring" className="nav-link" aria-current="page">
                      Monitoring
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

// export default Navbar;
