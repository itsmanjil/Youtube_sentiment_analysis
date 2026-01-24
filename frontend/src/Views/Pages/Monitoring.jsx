import React, { useContext, useEffect, useState } from "react";
import Sidenavbar from "../../Components/Sidenavbar";
import Fixedplugins from "../../Components/Fixedplugins";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { jwtDecode } from "jwt-decode";
import AuthContext from "../../context/AuthContext";

function Monitoring() {
  const navigate = useNavigate();
  const { authToken } = useContext(AuthContext);

  const [analyses, setAnalyses] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastRefreshTime, setLastRefreshTime] = useState(null);
  const [user, setUser] = useState({});

  // Fetch analyses from API
  const fetchAnalyses = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await axios({
        method: "GET",
        url: "http://127.0.0.1:8000/api/youtube/analyses/",
        timeout: 1000 * 10,
        validateStatus: (status) => {
          return status < 500;
        },
        headers: {
          Authorization: authToken
            ? "Bearer " + String(authToken.access)
            : null,
          "Content-Type": "application/json",
          accept: "application/json",
        },
      });

      if (response.status === 200 && response.data.data) {
        setAnalyses(response.data.data);
        setLastRefreshTime(new Date());
      } else {
        setError("Failed to load analyses");
      }
    } catch (err) {
      console.error("Error fetching analyses:", err);
      setError("Failed to load analyses. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Get user data
  const getUserData = async () => {
    try {
      const token = localStorage.getItem("authToken");
      const { user_id, user_name } = jwtDecode(token);
      setUser({ user_id, user_name });
    } catch (err) {
      console.error("Error getting user data:", err);
    }
  };

  // Fetch on component mount
  useEffect(() => {
    getUserData();
    fetchAnalyses();
  }, []);

  // Calculate summary statistics
  const calculateStats = () => {
    if (analyses.length === 0) {
      return {
        totalAnalyses: 0,
        avgPositive: 0,
        avgNegative: 0,
        avgNeutral: 0,
      };
    }

    let totalPositive = 0;
    let totalNegative = 0;
    let totalNeutral = 0;
    let totalComments = 0;

    analyses.forEach((analysis) => {
      const positive = analysis.sentiment_data?.Positive || 0;
      const negative = analysis.sentiment_data?.Negative || 0;
      const neutral = analysis.sentiment_data?.Neutral || 0;
      const total = positive + negative + neutral;

      if (total > 0) {
        totalPositive += (positive / total) * 100;
        totalNegative += (negative / total) * 100;
        totalNeutral += (neutral / total) * 100;
        totalComments++;
      }
    });

    return {
      totalAnalyses: analyses.length,
      avgPositive: totalComments > 0 ? (totalPositive / totalComments).toFixed(1) : 0,
      avgNegative: totalComments > 0 ? (totalNegative / totalComments).toFixed(1) : 0,
      avgNeutral: totalComments > 0 ? (totalNeutral / totalComments).toFixed(1) : 0,
    };
  };

  const stats = calculateStats();

  // Format timestamp to relative time
  const getRelativeTime = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);

    if (seconds < 60) return "Just now";
    if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} hours ago`;
    if (seconds < 2592000) return `${Math.floor(seconds / 86400)} days ago`;
    return date.toLocaleDateString();
  };

  // Navigate to dashboard with analysis data
  const viewAnalysisDetails = async (analysis) => {
    const fallbackState = {
      sentiment_data: analysis.sentiment_data,
      video: analysis.video,
      total_analyzed: analysis.total_comments_analyzed,
      model_used: analysis.analysis_model,
      like_weighted_sentiment: analysis.like_weighted_sentiment,
      top_words_positive: analysis.top_words_positive,
      top_words_negative: analysis.top_words_negative,
      filtered: analysis.filtered,
      confidence_stats: analysis.confidence_stats,
    };

    const videoId = analysis.video?.id;
    if (!videoId) {
      navigate("/dashboard", { state: fallbackState });
      return;
    }

    try {
      const response = await axios({
        method: "GET",
        url: `http://127.0.0.1:8000/api/youtube/analysis/${videoId}/`,
        timeout: 1000 * 10,
        validateStatus: (status) => {
          return status < 500;
        },
        headers: {
          Authorization: authToken
            ? "Bearer " + String(authToken.access)
            : null,
          "Content-Type": "application/json",
          accept: "application/json",
        },
      });

      if (response.status === 200 && response.data?.data) {
        const detail = response.data.data;
        navigate("/dashboard", {
          state: {
            sentiment_data: detail.sentiment_data,
            video: detail.video,
            total_analyzed: detail.total_comments,
            model_used: detail.model_used,
            like_weighted_sentiment: detail.like_weighted_sentiment,
            top_words_positive: detail.top_words_positive,
            top_words_negative: detail.top_words_negative,
            filtered: detail.filtered,
            sentiment_timeline: detail.sentiment_timeline,
            confidence_stats: detail.confidence_stats,
            sentiment_confidence_intervals: detail.sentiment_confidence_intervals,
            aspect_sentiment: detail.aspect_sentiment,
            analysis_meta: detail.analysis_meta,
            fetched_date: detail.fetched_date,
          },
        });
        return;
      }
    } catch (err) {
      console.error("Error fetching analysis detail:", err);
    }

    navigate("/dashboard", { state: fallbackState });
  };

  // Sentiment color coding
  const getSentimentColor = (sentiment) => {
    if (sentiment === "Positive") return "#008001";
    if (sentiment === "Negative") return "#FF0000";
    return "#0000FF";
  };

  // Get dominant sentiment
  const getDominantSentiment = (sentimentData) => {
    const positive = sentimentData?.Positive || 0;
    const negative = sentimentData?.Negative || 0;
    const neutral = sentimentData?.Neutral || 0;

    if (positive >= negative && positive >= neutral) return "Positive";
    if (negative >= positive && negative >= neutral) return "Negative";
    return "Neutral";
  };

  return (
    <>
      <Sidenavbar />
      <main className="main-content position-relative max-height-vh-100 h-100 border-radius-lg">
        <nav
          className="navbar navbar-main navbar-expand-lg px-0 shadow-none border-radius-xl"
          id="navbarBlur"
          data-scroll="true"
        >
          <div className="container-fluid py-1 px-3">
            <nav aria-label="breadcrumb">
              <h2 className="font-weight-bolder mb-0">Monitoring Dashboard</h2>
            </nav>

            <div
              className="collapse navbar-collapse mt-sm-0 mt-2 me-md-0 me-sm-4"
              id="navbar"
            >
              <div className="ms-md-auto pe-md-3 d-flex align-items-center">
                <div className="input-group input-group-outline">
                  <button
                    className="btn btn-light profile-button bg-primary"
                    type="button"
                    onClick={fetchAnalyses}
                    disabled={isLoading}
                    style={{
                      color: "white",
                      margin: 0,
                      textTransform: "capitalize",
                    }}
                  >
                    {isLoading ? "Refreshing..." : "Refresh"}
                  </button>
                </div>
                <div className="input-group input-group-outline" style={{ marginLeft: "10px" }}>
                  <Link to="/search">
                    <input
                      className="btn btn-light profile-button bg-primary"
                      type="button"
                      value="Analyze Video"
                      style={{
                        color: "white",
                        margin: 0,
                        textTransform: "capitalize",
                      }}
                    ></input>
                  </Link>
                </div>
              </div>

              <ul className="navbar-nav justify-content-end">
                <li className="nav-item d-flex align-items-center">
                  <Link
                    to="/profile"
                    className="nav-link text-body font-weight-bold px-0"
                  >
                    <i className="fa fa-user me-sm-1"></i>
                    <span className="d-sm-inline d-none">{user.user_name}</span>
                  </Link>
                </li>
                <li className="nav-item d-xl-none ps-3 d-flex align-items-center">
                  <button
                    type="button"
                    className="nav-link text-body p-0"
                    id="iconNavbarSidenav"
                    style={{ border: "none", background: "transparent" }}
                  >
                    <div className="sidenav-toggler-inner">
                      <i className="sidenav-toggler-line"></i>
                      <i className="sidenav-toggler-line"></i>
                      <i className="sidenav-toggler-line"></i>
                    </div>
                  </button>
                </li>
              </ul>
            </div>
          </div>
        </nav>

        <div className="container-fluid py-4">
          {error && (
            <div className="alert alert-danger alert-dismissible fade show" role="alert">
              {error}
              <button
                type="button"
                className="btn-close"
                onClick={() => setError(null)}
                aria-label="Close"
              ></button>
            </div>
          )}

          {lastRefreshTime && (
            <p className="text-sm text-muted mb-3">
              Last refreshed: {lastRefreshTime.toLocaleTimeString()}
            </p>
          )}

          {/* Summary Statistics */}
          <div className="row mb-4">
            <div className="col-xl-4 col-sm-6 mb-xl-0 mb-4">
              <div className="card">
                <div className="card-header p-3 pt-2">
                  <div className="text-left pt-1">
                    <p className="text-sm mb-0 text-capitalize">Total Analyses</p>
                    <h4 className="mb-0">{stats.totalAnalyses}</h4>
                  </div>
                </div>
                <hr className="dark horizontal my-0" />
              </div>
            </div>
            <div className="col-xl-4 col-sm-6 mb-xl-0 mb-4">
              <div className="card">
                <div className="card-header p-3 pt-2">
                  <div className="text-left pt-1">
                    <p className="text-sm mb-0 text-capitalize">Avg Positive %</p>
                    <h4 className="mb-0" style={{ color: "#008001" }}>
                      {stats.avgPositive}%
                    </h4>
                  </div>
                </div>
                <hr className="dark horizontal my-0" />
              </div>
            </div>
            <div className="col-xl-4 col-sm-6">
              <div className="card">
                <div className="card-header p-3 pt-2">
                  <div className="text-left pt-1">
                    <p className="text-sm mb-0 text-capitalize">Avg Negative %</p>
                    <h4 className="mb-0" style={{ color: "#FF0000" }}>
                      {stats.avgNegative}%
                    </h4>
                  </div>
                </div>
                <hr className="dark horizontal my-0" />
              </div>
            </div>
          </div>

          {/* Recent Analyses Feed */}
          <div className="row">
            <div className="col-12">
              <div className="card">
                <div className="card-header pb-0">
                  <h6>Recent Analyses</h6>
                </div>
                <div className="card-body">
                  {isLoading && analyses.length === 0 ? (
                    <div className="text-center py-4">
                      <div className="spinner-border text-primary" role="status">
                        <span className="sr-only">Loading...</span>
                      </div>
                      <p className="mt-2">Loading analyses...</p>
                    </div>
                  ) : analyses.length === 0 ? (
                    <div className="text-center py-4">
                      <i className="fas fa-chart-line" style={{ fontSize: "48px", color: "#ccc" }}></i>
                      <p className="text-muted mt-3 mb-3">
                        No analyses yet. Analyze your first YouTube video!
                      </p>
                      <Link to="/search">
                        <button className="btn btn-primary">Analyze Video</button>
                      </Link>
                    </div>
                  ) : (
                    <div className="table-responsive">
                      {analyses.map((analysis, index) => {
                        const positive = analysis.sentiment_data?.Positive || 0;
                        const negative = analysis.sentiment_data?.Negative || 0;
                        const neutral = analysis.sentiment_data?.Neutral || 0;
                        const total = positive + negative + neutral;

                        const positivePercent = total > 0 ? ((positive / total) * 100).toFixed(1) : 0;
                        const negativePercent = total > 0 ? ((negative / total) * 100).toFixed(1) : 0;
                        const neutralPercent = total > 0 ? ((neutral / total) * 100).toFixed(1) : 0;

                        const dominantSentiment = getDominantSentiment(analysis.sentiment_data);

                        return (
                          <div
                            key={analysis.id || index}
                            className="card mb-3"
                            style={{ boxShadow: "0 2px 4px rgba(0,0,0,0.1)" }}
                          >
                            <div className="card-body">
                              <div className="row align-items-center">
                                {/* Video Thumbnail */}
                                <div className="col-md-2 text-center">
                                  {analysis.video?.thumbnail_url ? (
                                    <img
                                      src={analysis.video.thumbnail_url}
                                      alt="Video thumbnail"
                                      style={{
                                        width: "100%",
                                        maxWidth: "120px",
                                        borderRadius: "8px",
                                      }}
                                    />
                                  ) : (
                                    <div
                                      style={{
                                        width: "120px",
                                        height: "90px",
                                        backgroundColor: "#e0e0e0",
                                        borderRadius: "8px",
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                      }}
                                    >
                                      <i className="fas fa-video" style={{ fontSize: "24px", color: "#999" }}></i>
                                    </div>
                                  )}
                                </div>

                                {/* Video Info */}
                                <div className="col-md-4">
                                  <h6 className="mb-1">
                                    {analysis.video?.title || "Untitled Video"}
                                  </h6>
                                  <p className="text-sm text-muted mb-1">
                                    <i className="fas fa-user-circle"></i>{" "}
                                    {analysis.video?.channel_name || "Unknown Channel"}
                                  </p>
                                  <p className="text-xs text-muted mb-0">
                                    <i className="fas fa-clock"></i>{" "}
                                    {getRelativeTime(analysis.fetched_date)}
                                  </p>
                                  <p className="text-xs text-muted mb-0">
                                    Model: {analysis.analysis_model || "LOGREG"}
                                  </p>
                                </div>

                                {/* Sentiment Stats */}
                                <div className="col-md-4">
                                  <p className="text-sm mb-2">
                                    <strong>Sentiment Breakdown:</strong>
                                  </p>
                                  <div className="mb-1">
                                    <span
                                      className="badge"
                                      style={{
                                        backgroundColor: "#008001",
                                        color: "white",
                                        marginRight: "5px",
                                      }}
                                    >
                                      Positive: {positive} ({positivePercent}%)
                                    </span>
                                  </div>
                                  <div className="mb-1">
                                    <span
                                      className="badge"
                                      style={{
                                        backgroundColor: "#FF0000",
                                        color: "white",
                                        marginRight: "5px",
                                      }}
                                    >
                                      Negative: {negative} ({negativePercent}%)
                                    </span>
                                  </div>
                                  <div className="mb-1">
                                    <span
                                      className="badge"
                                      style={{
                                        backgroundColor: "#0000FF",
                                        color: "white",
                                        marginRight: "5px",
                                      }}
                                    >
                                      Neutral: {neutral} ({neutralPercent}%)
                                    </span>
                                  </div>
                                  <p className="text-xs text-muted mt-2">
                                    Total Comments: {analysis.total_comments_analyzed || total}
                                  </p>
                                </div>

                                {/* Action Button */}
                                <div className="col-md-2 text-center">
                                  <div
                                    className="mb-2"
                                    style={{
                                      padding: "10px",
                                      borderRadius: "8px",
                                      backgroundColor: getSentimentColor(dominantSentiment) + "20",
                                    }}
                                  >
                                    <p className="text-xs mb-0" style={{ fontWeight: "bold" }}>
                                      Overall
                                    </p>
                                    <p
                                      className="text-sm mb-0"
                                      style={{
                                        color: getSentimentColor(dominantSentiment),
                                        fontWeight: "bold",
                                      }}
                                    >
                                      {dominantSentiment}
                                    </p>
                                  </div>
                                  <button
                                    className="btn btn-sm btn-primary"
                                    onClick={() => viewAnalysisDetails(analysis)}
                                  >
                                    View Details
                                  </button>
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
      <Fixedplugins />
    </>
  );
}

export default Monitoring;
