import React from "react";
import "./Homepage.css";
import { Link } from "react-router-dom";
import { HashLink } from "react-router-hash-link";
import Navbar from "../../Components/Navbar";
// css from StyleSheet.css - bg color change garnu pare change from there
function Homepage() {
  const token = localStorage.getItem("authToken");
  return (
    <div data-bs-spy="scroll" data-bs-target="#navbarExample">
      {/* <!-- Navigation --> */}
      <Navbar />

      {/* <!-- Header --> */}
      <header id="header" className="header">
        <div className="container">
          <div className="row">
            <div className="col-lg-6">
              <div className="text-container">
                <h1 className="h1-large">
                  YouTube Sentiment Analysis Tool for Smart Insights!
                </h1>
                <p className="p-large">
                  Discover what viewers really think about YouTube videos through advanced sentiment analysis.
                </p>
                {token == null && (
                  <Link to="/register" className="btn-solid-lg">
                    Sign up for free
                  </Link>
                )}
                {token !== null && (
                  <HashLink smooth to="#features" className="btn-solid-lg">
                    Learn More!
                  </HashLink>
                )}
              </div>
            </div>
            <div className="col-lg-6">
              <div className="image-container">
                <img
                  className="img-fluid"
                  src="../assets/img/header-illustration.svg"
                  alt="alternative"
                />
              </div>
            </div>
          </div>
        </div>
      </header>
      {/* <!-- Features --> */}
      <div id="features" className="cards-1">
        <div className="container">
          <div className="row">
            <div className="col-lg-12">
              <h2 className="h2-heading">
                YouTube Sentiment Analysis application is packed with{" "}
                <span>awesome features</span>
              </h2>
            </div>
          </div>
          <div className="row">
            <div className="col-lg-12">
              <div className="card">
                <div className="card-icon">
                  <span className="fas fa-headphones-alt"></span>
                </div>
                <div className="card-body">
                  <h4 className="card-title">Viewer Sentiments</h4>
                  <p>Analyze viewer opinions and emotions from YouTube video comments.</p>
                </div>
              </div>
              <div className="card">
                <div className="card-icon green">
                  <span className="far fa-clipboard"></span>
                </div>
                <div className="card-body">
                  <h4 className="card-title">Engagement Tracking</h4>
                  <p>Track video engagement and sentiment trends over time.</p>
                </div>
              </div>
              <div className="card">
                <div className="card-icon blue">
                  <span className="far fa-comments"></span>
                </div>
                <div className="card-body">
                  <h4 className="card-title">Reporting Tool</h4>
                  <p>Download comprehensive sentiment analysis reports for YouTube videos.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* <!-- Details 1 --> */}
      <div id="details" className="basic-1 bg-gray">
        <div className="container">
          <div className="row">
            <div className="col-lg-6 col-xl-5">
              <div className="text-container">
                <h2>
                  Understand viewer sentiment and improve your content strategy
                </h2>
                <p>
                  Leverage powerful AI-driven sentiment analysis to decode viewer emotions and opinions from YouTube comments. Make data-driven decisions to enhance your content and build stronger audience connections.
                </p>
              </div>
            </div>
            <div className="col-lg-6 col-xl-7">
              <div className="image-container">
                <img
                  className="img-fluid"
                  src="../assets/img/details-1.svg"
                  alt="alternative"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* <!-- Invitation --> */}
      
      {/* <!-- Pricing --> */}
      
      {/* <!-- Footer --> */}
      <div className="footer">
        <div className="container">
          <div className="row">
            <div className="col-lg-12">
              <div className="footer-col first">
                <h6>About Website</h6>
                <p className="p-small">
                  YouTube Sentiment Analysis helps you understand what viewers really think about videos through intelligent comment analysis.{" "}
                </p>
              </div>
              <div className="footer-col second"></div>
              <div className="footer-col third">
                <span className="fa-stack">
                  <a href="#your-link">
                    <i className="fas fa-circle fa-stack-2x"></i>
                    <i className="fab fa-facebook-f fa-stack-1x"></i>
                  </a>
                </span>
                <span className="fa-stack">
                  <a href="#your-link">
                    <i className="fas fa-circle fa-stack-2x"></i>
                    <i className="fab fa-reddit fa-stack-1x"></i>
                  </a>
                </span>
                <span className="fa-stack">
                  <a href="#your-link">
                    <i className="fas fa-circle fa-stack-2x"></i>
                    <i className="fab fa-pinterest-p fa-stack-1x"></i>
                  </a>
                </span>
                <span className="fa-stack">
                  <a href="#your-link">
                    <i className="fas fa-circle fa-stack-2x"></i>
                    <i className="fab fa-instagram fa-stack-1x"></i>
                  </a>
                </span>
                <p className="p-small">
                  For further queries please contact us at:{" "}
                  <a href="mailto:contact@site.com">
                    <strong>youtubesentiment@site.com</strong>
                  </a>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* <!-- Copyright --> */}
      <div className="copyright" id="contact">
        <div className="container">
          <div className="row">
            <div className="col-lg-6">
              <p className="p-small">
                Copyright © <a href="#your-link">YouTube Sentiment Analysis</a>
              </p>
            </div>

            <div className="col-lg-6">
              <p className="p-small">
                Distributed By
                <a href="https://themewagon.com/"> Business Analytics</a>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Homepage;
