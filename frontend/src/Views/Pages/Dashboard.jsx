import React, { useContext, useEffect, useState, useRef } from "react";
//kun component lai k chainxa tei line   send a data from a component to all child components.

//focusing on emmail in login

import { useNavigate } from "react-router-dom";
import { useCurrentPng } from "recharts-to-png";
import Sidenavbar from "../../Components/Sidenavbar";
import Fixedplugins from "../../Components/Fixedplugins";
import { Link, useLocation } from "react-router-dom";
import axios from "axios";
import FileSaver from "file-saver";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ComposedChart,
  ScatterChart,
  Scatter,
} from "recharts";
import axiosInstance from "../../axios";
import { jwtDecode } from "jwt-decode";
import AuthContext from "../../context/AuthContext";

const data = [
  {
    name: "Page A", //year,month ->Time
    positive: 4000,
    negative: 2400,
    neutral: 2400,
  },
  {
    name: "Page B",
    positive: 3000,
    negative: 1398,
    neutral: 2210,
  },
  {
    name: "Page C",
    positive: 2000,
    negative: 9800,
    neutral: 2290,
  },
  {
    name: "Page D",
    positive: 2780,
    negative: 3908,
    neutral: 2000,
  },
  {
    name: "Page E",
    positive: 1890,
    negative: 4800,
    neutral: 2181,
  },
  {
    name: "Page F",
    positive: 2390,
    negative: 3800,
    neutral: 2500,
  },
  {
    name: "Page G",
    positive: 3490,
    negative: 4300,
    neutral: 2100,
  },
  {
    name: "Page H",
    positive: 3490,
    negative: 4300,
    neutral: 2100,
  },
  {
    name: "Page I",
    positive: 3490,
    negative: 4300,
    neutral: 2100,
  },
];
// Removed unused variable data01
// custom label for pie chart
const COLORS = ["#FF0000", "#0000FF", "#008001"];

const RADIAN = Math.PI / 180;
const renderCustomizedLabel = ({
  cx,
  cy,
  midAngle,
  innerRadius,
  outerRadius,
  percent,
  index,
}) => {
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return (
    <text
      x={x}
      y={y}
      fill="white"
      textAnchor={x > cx ? "start" : "end"}
      dominantBaseline="central"
    >
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};
function Dashboard(props) {
  const navigate = useNavigate();
  const location = useLocation();
  const { authToken } = useContext(AuthContext);

  function refresh() {}
  function calendar() {}
  const [user, setUser] = useState({});
  const [sentimentBreakdown, setSentimentBreakdown] = useState();
  const [videoData, setVideoData] = useState();
  const [hourdata, setHourdata] = useState();
  const [fetchedDate, setFetchedDate] = useState();
  const [videoTitle, setVideoTitle] = useState();
  const [hasSearched, setHasSearched] = useState(false);
  const [grpahState, setGraphState] = useState(false);
  const [searchedList, setSearchedList] = useState([]);
  const [topWordsPositive, setTopWordsPositive] = useState([]);
  const [topWordsNegative, setTopWordsNegative] = useState([]);
  const [topComments, setTopComments] = useState([]);
  const [filterStats, setFilterStats] = useState(null);
  const [confidenceStats, setConfidenceStats] = useState(null);
  const [confidenceIntervals, setConfidenceIntervals] = useState(null);
  const [aspectSentiment, setAspectSentiment] = useState([]);
  const [analysisMeta, setAnalysisMeta] = useState(null);
  const [modelUsed, setModelUsed] = useState(null);
  const [userStats, setUserStats] = useState({
    totalVideos: 0,
    totalComments: 0,
    avgPositive: 0,
    avgNegative: 0,
  });

  const formatPercent = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return "0%";
    }
    return `${(value * 100).toFixed(1)}%`;
  };

  const ensembleInfo = analysisMeta?.ensemble;
  const ensembleModels = ensembleInfo?.models || [];
  const ensembleWeights = ensembleInfo?.weights || {};
  const ensembleWeightEntries = Object.entries(ensembleWeights);
  const metaLearnerInfo = analysisMeta?.meta_learner;
  const fuzzyInfo = analysisMeta?.fuzzy;

  const getData = async () => {
    try {
      const token = localStorage.getItem("authToken");
      const { user_id, user_name } = jwtDecode(token);
      console.log(user_name);
      if (user_id) {
        const userDatas = await axios({
          method: "GET",
          url: `http://127.0.0.1:8000/api/user/me/${user_id}`,
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
        setUser({
          user_name: userDatas.data.user_name,
          email: userDatas.data.email,
        });
        const list = Array.isArray(userDatas.data.searched_list)
          ? userDatas.data.searched_list
          : [];
        setSearchedList(list);
        console.log("user", user);

        // Fetch all user's analyses for statistics
        try {
          const analysesResponse = await axios({
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

          if (analysesResponse.status === 200 && analysesResponse.data.data) {
            const analyses = analysesResponse.data.data;

            // Calculate aggregated statistics
            let totalVideos = analyses.length;
            let totalComments = 0;
            let totalPositive = 0;
            let totalNegative = 0;
            let totalNeutral = 0;

            analyses.forEach((analysis) => {
              const positive = analysis.sentiment_data?.Positive || 0;
              const negative = analysis.sentiment_data?.Negative || 0;
              const neutral = analysis.sentiment_data?.Neutral || 0;
              const total = positive + negative + neutral;

              totalComments += total;
              totalPositive += positive;
              totalNegative += negative;
              totalNeutral += neutral;
            });

            const avgPositive = totalComments > 0 ? ((totalPositive / totalComments) * 100).toFixed(1) : 0;
            const avgNegative = totalComments > 0 ? ((totalNegative / totalComments) * 100).toFixed(1) : 0;

            setUserStats({
              totalVideos,
              totalComments,
              avgPositive,
              avgNegative,
            });
          }
        } catch (err) {
          console.log("Error fetching user analyses:", err);
        }

        // Check if YouTube analysis data was passed from Search page
        if (location.state && location.state.sentiment_data) {
          const youtubeData = location.state;
          console.log("YouTube data received:", youtubeData);

          const sentimentArray = [
            { name: "Negative", value: youtubeData.sentiment_data.Negative || 0 },
            { name: "Neutral", value: youtubeData.sentiment_data.Neutral || 0 },
            { name: "Positive", value: youtubeData.sentiment_data.Positive || 0 },
          ];

          setSentimentBreakdown(sentimentArray);
          setVideoData(youtubeData.video);
          setVideoTitle(youtubeData.video?.title || "YouTube Video");
          setFetchedDate(
            youtubeData.fetched_date
              ? new Date(youtubeData.fetched_date).toLocaleDateString()
              : new Date().toLocaleDateString()
          );
          setHasSearched(true);

          // Set word cloud data (top words for positive and negative comments)
          if (youtubeData.top_words_positive) {
            setTopWordsPositive(youtubeData.top_words_positive);
          }
          if (youtubeData.top_words_negative) {
            setTopWordsNegative(youtubeData.top_words_negative);
          }

          // Set top comments (like-weighted sentiment)
          if (youtubeData.like_weighted_sentiment) {
            setTopComments(youtubeData.like_weighted_sentiment);
          }

          // Set filter statistics
          if (youtubeData.filtered) {
            setFilterStats(youtubeData.filtered);
          }

          const timelineSource =
            youtubeData.sentiment_timeline ||
            youtubeData.hour_data ||
            youtubeData.hourdata;
          if (timelineSource && typeof timelineSource === "object") {
            const timelineEntries = Object.entries(timelineSource)
              .map(([time, counts]) => ({ time, counts }))
              .sort((a, b) => new Date(a.time) - new Date(b.time))
              .map(({ time, counts }) => ({
                time: new Date(time).toLocaleString(),
                positive: counts.Positive || 0,
                negative: counts.Negative || 0,
                neutral: counts.Neutral || 0,
              }));

            if (timelineEntries.length > 0) {
              setHourdata(timelineEntries);
              setGraphState(true);
            } else {
              setHourdata(null);
              setGraphState(false);
            }
          } else {
            setHourdata(null);
            setGraphState(false);
          }

          if (youtubeData.confidence_stats) {
            setConfidenceStats(youtubeData.confidence_stats);
          }
          if (youtubeData.sentiment_confidence_intervals) {
            setConfidenceIntervals(youtubeData.sentiment_confidence_intervals);
          }
          if (youtubeData.aspect_sentiment) {
            setAspectSentiment(youtubeData.aspect_sentiment);
          }
          if (youtubeData.analysis_meta) {
            setAnalysisMeta(youtubeData.analysis_meta);
          }
          if (youtubeData.model_used) {
            setModelUsed(youtubeData.model_used);
          }
        }
      }
    } catch (err) {
      console.log(err);
      setHasSearched(false);
    }
  };
  useEffect(() => {
    getData();
  }, [location.state]);

  //line graph
  let [getLinePng, { ref: lineRef }] = useCurrentPng();

  const lineDownload = React.useCallback(async () => {
    const png = await getLinePng();
    // console.log("clicked", lineRef);
    if (png) {
      // Download with FileSaver
      FileSaver.saveAs(png, "line-chart.png");
    }
  }, [getLinePng]);

  //pie-chart
  let [getPiePng, { ref: pieRef }] = useCurrentPng();

  const pieDownload = React.useCallback(async () => {
    const png = await getPiePng();
    // console.log("clicked", pieRef);
    if (png) {
      // Download with FileSaver
      FileSaver.saveAs(png, "pie-chart.png");
    }
  }, [getPiePng]);

  const navigateToReport = () => {
    const token = localStorage.getItem("authToken");
    const { user_id, user_name } = jwtDecode(token);
    navigate(`/report/${videoTitle}`, {
      state: {
        user_name,
        sentimentBreakdown,
        sentimentTimeline: hourdata,
        fetchedDate,
        videoTitle,
        confidenceStats,
        confidenceIntervals,
        aspectSentiment,
        analysisMeta,
        modelUsed,
      },
    });
  };

  return (
    <>
      <Sidenavbar />
      <main className="main-content position-relative max-height-vh-100 h-100 border-radius-lg ">
        <nav
          className="navbar navbar-main navbar-expand-lg px-0 shadow-none border-radius-xl"
          id="navbarBlur"
          data-scroll="true"
        >
          <div className="container-fluid py-1 px-3">
            <nav aria-label="breadcrumb">
              <h2 className="font-weight-bolder mb-0">Dashboard</h2>
            </nav>

            <div
              className="collapse navbar-collapse mt-sm-0 mt-2 me-md-0 me-sm-4"
              id="navbar"
            >
              <div className="ms-md-auto pe-md-3 d-flex align-items-center">
                

                <div className="input-group input-group-outline">
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
              {hasSearched && (
                <div className="dropdown float-lg-end pe-4">
                  <button
                    onClick={navigateToReport}
                    className="fas fa-file"
                    style={{ border: "none", background: "transparent" }}
                  >
                    <span style={{ marginLeft: "5px" }}>Report</span>
                  </button>
                </div>
              )}
              

              <ul className="navbar-nav  justify-content-end">
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
                
                
                
                {/* </li> */}
              </ul>
            </div>
          </div>
        </nav>

        {/* User Statistics Overview */}
        {userStats.totalVideos > 0 && !hasSearched && (
          <div className="container-fluid py-4">
            <div className="row mb-3">
              <div className="col-12">
                <div className="card" style={{ backgroundColor: "#f8f9fa", border: "2px solid #e9ecef" }}>
                  <div className="card-header pb-0" style={{ backgroundColor: "transparent" }}>
                    <div className="row align-items-center">
                      <div className="col">
                        <h6 className="mb-0">
                          <i className="fas fa-chart-line" style={{ color: "#3498db", marginRight: "8px" }}></i>
                          Your Analysis Overview
                        </h6>
                        <p className="text-xs text-muted mb-0 mt-1">
                          Aggregated statistics from all your YouTube video analyses
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="card-body pt-3">
                    <div className="row">
                      <div className="col-xl-3 col-md-6 mb-3">
                        <div className="text-center p-3" style={{ backgroundColor: "white", borderRadius: "8px" }}>
                          <i className="fas fa-video" style={{ fontSize: "28px", color: "#e74c3c" }}></i>
                          <h3 className="mt-2 mb-0">{userStats.totalVideos}</h3>
                          <p className="text-sm text-muted mb-0">Videos Analyzed</p>
                        </div>
                      </div>
                      <div className="col-xl-3 col-md-6 mb-3">
                        <div className="text-center p-3" style={{ backgroundColor: "white", borderRadius: "8px" }}>
                          <i className="fas fa-comments" style={{ fontSize: "28px", color: "#3498db" }}></i>
                          <h3 className="mt-2 mb-0">{userStats.totalComments.toLocaleString()}</h3>
                          <p className="text-sm text-muted mb-0">Total Comments</p>
                        </div>
                      </div>
                      <div className="col-xl-3 col-md-6 mb-3">
                        <div className="text-center p-3" style={{ backgroundColor: "white", borderRadius: "8px" }}>
                          <i className="fas fa-smile" style={{ fontSize: "28px", color: "#2ecc71" }}></i>
                          <h3 className="mt-2 mb-0" style={{ color: "#2ecc71" }}>{userStats.avgPositive}%</h3>
                          <p className="text-sm text-muted mb-0">Avg Positive</p>
                        </div>
                      </div>
                      <div className="col-xl-3 col-md-6 mb-3">
                        <div className="text-center p-3" style={{ backgroundColor: "white", borderRadius: "8px" }}>
                          <i className="fas fa-frown" style={{ fontSize: "28px", color: "#e74c3c" }}></i>
                          <h3 className="mt-2 mb-0" style={{ color: "#e74c3c" }}>{userStats.avgNegative}%</h3>
                          <p className="text-sm text-muted mb-0">Avg Negative</p>
                        </div>
                      </div>
                    </div>
                    <div className="row mt-2">
                      <div className="col-12 text-center">
                        <p className="text-xs text-muted mb-0">
                          <i className="fas fa-info-circle"></i> These statistics are calculated across all your analyzed videos.
                          <Link to="/monitoring" style={{ marginLeft: "8px", textDecoration: "none" }}>
                            View detailed history →
                          </Link>
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {!hasSearched && userStats.totalVideos === 0 && (
          <div className="container-fluid py-4">
            <div className="row">
              <div className="col-12">
                <div className="card text-center" style={{ padding: "40px" }}>
                  <div className="card-body">
                    <i className="fas fa-chart-bar" style={{ fontSize: "64px", color: "#ccc", marginBottom: "20px" }}></i>
                    <h4>No analysis yet</h4>
                    <p className="text-muted">Click "Analyze Video" to get started with your first YouTube sentiment analysis.</p>
                    <Link to="/search">
                      <button className="btn btn-primary mt-3">
                        <i className="fas fa-play-circle"></i> Analyze Your First Video
                      </button>
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        {hasSearched && (
          <div className="container-fluid py-4">
            <div className="row">
              <h6>Analysis results for: {videoTitle}</h6>
              {videoData && (
                <p style={{ fontSize: "14px", color: "#666" }}>
                  Channel: {videoData.channel} | Views: {videoData.view_count?.toLocaleString()} | Likes: {videoData.like_count?.toLocaleString()}
                </p>
              )}
              <div className="col-xl-3 col-sm-6 mb-xl-0 mb-4">
                <div className="card">
                  <div className="card-header p-3 pt-2">

                    <div className="text-left pt-1">
                      <p className="text-sm mb-0 text-capitalize">
                        Total Comments
                      </p>
                      <h4 className="mb-0">
                        {sentimentBreakdown && sentimentBreakdown.length >= 3
                          ? (sentimentBreakdown[0]?.value || 0) +
                            (sentimentBreakdown[1]?.value || 0) +
                            (sentimentBreakdown[2]?.value || 0)
                          : 0}
                      </h4>
                      {/* pull total comments data into this h4 */}
                    </div>
                  </div>
                  <hr className="dark horizontal my-0" />

                </div>
              </div>
              <div className="col-xl-3 col-sm-6 mb-xl-0 mb-4">
                <div className="card">
                  <div className="card-header p-3 pt-2">

                    <div className="text-left pt-1">
                      <p className="text-sm mb-0 text-capitalize">
                        Positive Comments
                      </p>
                      <h4 className="mb-0">{sentimentBreakdown && sentimentBreakdown[2] ? sentimentBreakdown[2].value : 0}</h4>
                      {/* pull positive comments data into this h4 */}
                    </div>
                  </div>
                  <hr className="dark horizontal my-0" />

                </div>
              </div>
              <div className="col-xl-3 col-sm-6 mb-xl-0 mb-4">
                <div className="card">
                  <div className="card-header p-3 pt-2">

                    <div className="text-left pt-1">
                      <p className="text-sm mb-0 text-capitalize">
                        Negative Comments
                      </p>
                      <h4 className="mb-0">{sentimentBreakdown && sentimentBreakdown[0] ? sentimentBreakdown[0].value : 0}</h4>
                      {/* pull negative comments data into this h4 */}
                    </div>
                  </div>
                  <hr className="dark horizontal my-0" />

                </div>
              </div>
              <div className="col-xl-3 col-sm-6">
                <div className="card">
                  <div className="card-header p-3 pt-2">

                    <div className="text-left pt-1">
                      <p className="text-sm mb-0 text-capitalize">
                        Neutral Comments
                      </p>
                      <h4 className="mb-0">{sentimentBreakdown && sentimentBreakdown[1] ? sentimentBreakdown[1].value : 0}</h4>
                    </div>
                  </div>
                  <hr className="dark horizontal my-0" />

                </div>
              </div>
            </div>
            {(confidenceStats || analysisMeta || modelUsed) && (
              <div className="row mb-4">
                <div className="col-lg-6 col-md-6 mb-md-0 mb-4">
                  <div className="card h-100">
                    <div className="card-header pb-0">
                      <h6>Model & Experiment Settings</h6>
                    </div>
                    <div className="card-body">
                      <p className="text-sm mb-1">
                        <strong>Model Used:</strong> {modelUsed || "N/A"}
                      </p>
                      {ensembleModels.length > 0 && (
                        <p className="text-sm mb-1">
                          <strong>Ensemble Models:</strong> {ensembleModels.join(", ")}
                        </p>
                      )}
                      {ensembleWeightEntries.length > 0 && (
                        <div className="text-sm mb-2">
                          <strong>Ensemble Weights:</strong>
                          <ul className="mb-0">
                            {ensembleWeightEntries.map(([key, value]) => (
                              <li key={key}>
                                {key}: {value}
                              </li>
                            ))}
                          </ul>
                          {ensembleInfo?.weights_source && (
                            <p className="text-xs text-muted mb-0">
                              Weights source: {ensembleInfo.weights_source}
                            </p>
                          )}
                        </div>
                      )}
                      {metaLearnerInfo && (
                        <div className="text-sm mb-2">
                          <strong>Meta-Learner:</strong>
                          <ul className="mb-0">
                            {metaLearnerInfo.model_path && (
                              <li>Model Path: {metaLearnerInfo.model_path}</li>
                            )}
                            {metaLearnerInfo.base_models && (
                              <li>
                                Base Models: {metaLearnerInfo.base_models.join(", ")}
                              </li>
                            )}
                            {metaLearnerInfo.feature_type && (
                              <li>Feature Type: {metaLearnerInfo.feature_type}</li>
                            )}
                            {metaLearnerInfo.meta_learner_type && (
                              <li>Type: {metaLearnerInfo.meta_learner_type}</li>
                            )}
                          </ul>
                        </div>
                      )}
                      {fuzzyInfo && (
                        <div className="text-sm mb-2">
                          <strong>Fuzzy Configuration:</strong>
                          <ul className="mb-0">
                            {fuzzyInfo.base_models && (
                              <li>Base Models: {fuzzyInfo.base_models.join(", ")}</li>
                            )}
                            {fuzzyInfo.mf_type && <li>MF Type: {fuzzyInfo.mf_type}</li>}
                            {fuzzyInfo.defuzz_method && (
                              <li>Defuzz: {fuzzyInfo.defuzz_method}</li>
                            )}
                            {fuzzyInfo.t_norm && <li>T-Norm: {fuzzyInfo.t_norm}</li>}
                            {fuzzyInfo.t_conorm && (
                              <li>T-Conorm: {fuzzyInfo.t_conorm}</li>
                            )}
                            {fuzzyInfo.alpha_cut !== undefined && (
                              <li>Alpha Cut: {fuzzyInfo.alpha_cut}</li>
                            )}
                          </ul>
                        </div>
                      )}
                      {analysisMeta?.bootstrap_samples !== undefined && (
                        <p className="text-sm mb-1">
                          <strong>Bootstrap Samples:</strong> {analysisMeta.bootstrap_samples}
                        </p>
                      )}
                      {analysisMeta?.random_seed !== undefined && (
                        <p className="text-sm mb-0">
                          <strong>Random Seed:</strong> {analysisMeta.random_seed}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
                {confidenceStats && (
                  <div className="col-lg-6 col-md-6">
                    <div className="card h-100">
                      <div className="card-header pb-0">
                        <h6>Confidence & Uncertainty</h6>
                      </div>
                      <div className="card-body">
                        <div className="row text-center">
                          <div className="col-4">
                            <h4 className="mb-0">{formatPercent(confidenceStats.mean)}</h4>
                            <p className="text-xs text-muted mb-0">Mean</p>
                          </div>
                          <div className="col-4">
                            <h4 className="mb-0">{formatPercent(confidenceStats.median)}</h4>
                            <p className="text-xs text-muted mb-0">Median</p>
                          </div>
                          <div className="col-4">
                            <h4 className="mb-0">{formatPercent(confidenceStats.low_confidence_ratio)}</h4>
                            <p className="text-xs text-muted mb-0">Low Confidence</p>
                          </div>
                        </div>
                        {confidenceStats.threshold !== undefined && (
                          <p className="text-xs text-muted mt-3 mb-0">
                            Low-confidence threshold: {confidenceStats.threshold}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
            <br />
            {sentimentBreakdown && (
              <div className="row mb-4">
                <div className="col-lg-8 col-md-6 mb-md-0 mb-4">
                  <div className="card">
                    <div className="card-header pb-0">
                      <div className="row">
                        <div className="col-lg-6 col-7">
                          <h6>Sentiment Breakdown</h6>
                        </div>
                      </div>
                    </div>
                    <div className="card-body px-0 pb-2">
                      <div className="chart">
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart
                            data={sentimentBreakdown}
                            margin={{
                              top: 5,
                              right: 30,
                              left: 20,
                              bottom: 5,
                            }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="value">
                              {sentimentBreakdown.map((entry, index) => (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={COLORS[index % COLORS.length]}
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="col-lg-4 col-md-6">
                  <div className="card h-100">
                    <div className="card-header pb-0">
                      <h6>Search History</h6>
                    </div>
                    <div className="card-body p-3">
                      {searchedList.length === 0 ? (
                        <p className="text-sm mb-0">
                          No search history yet.
                        </p>
                      ) : (
                        <>
                          <h6 className="text-uppercase text-body text-xs font-weight-bolder">
                            You searched for:
                          </h6>
                          <ul className="list-group">
                            {searchedList.map((item, index) => {
                              const label =
                                typeof item === "string"
                                  ? item
                                  : item && typeof item === "object"
                                  ? item.title || item.video_id
                                  : "";
                              const keyBase =
                                item && typeof item === "object"
                                  ? item.video_id || item.title || "item"
                                  : label || "item";
                              return (
                                <li
                                  key={`${keyBase}-${index}`}
                                  className="list-group-item border-0 ps-0 pt-0 text-sm"
                                >
                                  <div className="text-dark">{label}</div>
                                </li>
                              );
                            })}
                          </ul>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
            {confidenceIntervals && (
              <div className="row mb-4">
                <div className="col-12">
                  <div className="card">
                    <div className="card-header pb-0">
                      <h6>Sentiment Confidence Intervals (95%)</h6>
                      <p className="text-sm text-muted">
                        Bootstrap confidence bounds for sentiment ratios.
                      </p>
                    </div>
                    <div className="card-body">
                      <div className="table-responsive">
                        <table className="table align-items-center mb-0">
                          <thead>
                            <tr>
                              <th>Sentiment</th>
                              <th>Lower</th>
                              <th>Upper</th>
                            </tr>
                          </thead>
                          <tbody>
                            {["Positive", "Neutral", "Negative"].map((label) => {
                              const interval = confidenceIntervals[label] || {};
                              return (
                                <tr key={label}>
                                  <td>{label}</td>
                                  <td>{formatPercent(interval.lower)}</td>
                                  <td>{formatPercent(interval.upper)}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            {Array.isArray(analysisMeta?.model_comparison) && analysisMeta.model_comparison.length > 0 && (
              <div className="row mb-4">
                <div className="col-12">
                  <div className="card">
                    <div className="card-header pb-0">
                      <h6>Model Comparison (Research Mode)</h6>
                      <p className="text-sm text-muted mb-0">
                        Summary metrics for CI experiments.
                      </p>
                    </div>
                    <div className="card-body">
                      <div className="table-responsive">
                        <table className="table align-items-center mb-0">
                          <thead>
                            <tr>
                              <th>Model</th>
                              <th>Accuracy</th>
                              <th>Macro F1</th>
                            </tr>
                          </thead>
                          <tbody>
                            {analysisMeta.model_comparison.map((row, index) => (
                              <tr key={index}>
                                <td>{row.name || row.model}</td>
                                <td>{row.accuracy}</td>
                                <td>{row.macro_f1}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            {hourdata && (
              <div className="row mb-4">
                <div className="col-lg-8 col-md-6 mb-md-0 mb-4">
                  <div className="card">
                    <div className="card-header pb-0">
                      <div className="row">
                        <div className="col-lg-6 col-7">
                          <p>Sentiment Timeline (hourly)</p>
                          {!grpahState && (
                            <span style={{ color: "red" }}>
                              No timeline data available for {fetchedDate}
                            </span>
                          )}
                        </div>
                        <div className="col-lg-6 col-5 my-auto text-end">
                          <div className="dropdown float-lg-end">
                            <button
                              onClick={lineDownload}
                              className="fas fa-download"
                              style={{
                                margin: "15px",
                                border: "none",
                                background: "transparent",
                              }}
                            ></button>
                            
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="card-body px-0 pb-2">
                      <div className="chart">
                        <ResponsiveContainer width="100%" height={370}>
                          <LineChart
                            ref={lineRef}
                            id="chart-bars"
                            className="chart-canvas"
                            width={700}
                            height={320}
                            data={hourdata}
                            margin={{
                              top: 5,
                              right: 30,
                              left: 20,
                              bottom: 5,
                            }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="time"
                              label={{
                                value: "Hour",
                                position: "bottom",
                              }}
                              scaleToFit="true"
                              verticalAnchor="start"
                              textAnchor="end"
                              interval={0}
                              angle="-18"
                              tick={{ fontSize: "12px" }}
                              // axisLine={false}
                            />
                            <YAxis
                              label={{
                                value: "No. of Comments",
                                angle: -90,
                                position: "insideLeft",
                              }}
                            />
                            <Tooltip />
                            <Legend
                              verticalAlign="top"
                              align="right"
                              wrapperStyle={{
                                left: 0,
                                top: -20,
                                paddingBottom: -60,
                              }}
                            />
                            <Line
                              type="monotone"
                              dataKey="positive"
                              stroke="#008001"
                              activeDot={{ r: 8 }}
                            />
                            <Line
                              type="monotone"
                              dataKey="negative"
                              stroke="#FF0000"
                              activeDot={{ r: 8 }}
                            />
                            <Line
                              type="monotone"
                              dataKey="neutral"
                              stroke="#0000FF"
                              activeDot={{ r: 8 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                        <p
                          style={{
                            textAlignVertical: "center",
                            textAlign: "center",
                          }}
                        >
                          Time
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="col-lg-4 col-md-6">
                  <div className="card h-100">
                    <div className="card-header pb-0">
                      <div className="row">
                        <div className="col-lg-6 col-7">
                          <h6>Types of Emotion</h6>
                        </div>
                        <div className="col-lg-6 col-5 my-auto text-end">
                          <div className="float-lg-end">
                            <button
                              onClick={pieDownload}
                              className="fas fa-download"
                              style={{
                                margin: "15px",
                                border: "none",
                                background: "transparent",
                              }}
                            ></button>
                            
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="card-body p-3">
                      <div className="chart">
                        <ResponsiveContainer width="100%" height={250}>
                          {sentimentBreakdown && (
                            <PieChart width={200} height={250} ref={pieRef}>
                              <Pie
                                dataKey="value"
                                isAnimationActive={false}
                                data={sentimentBreakdown}
                                outerRadius={80}
                                labelLine={false}
                                fill="#8884d8"
                                label={renderCustomizedLabel}
                              >
                                {data.map((entry, index) => (
                                  <Cell
                                    key={`cell-${index}`}
                                    fill={COLORS[index % COLORS.length]}
                                  />
                                ))}
                              </Pie>

                              <Tooltip />
                            </PieChart>
                          )}
                        </ResponsiveContainer>
                        <div>
                          <ul>
                            <ol>
                              <i
                                className="fas fa-circle"
                                style={{ color: "#008001" }}
                              ></i>{" "}
                              : Positive
                            </ol>
                            <ol>
                              <i
                                className="fas fa-circle"
                                style={{ color: "#FF0000" }}
                              ></i>{" "}
                              : Negative
                            </ol>
                            <ol>
                              <i
                                className="fas fa-circle"
                                style={{ color: "#0000FF" }}
                              ></i>{" "}
                              : Neutral
                            </ol>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {aspectSentiment && aspectSentiment.length > 0 && (
              <div className="row mb-4">
                <div className="col-12">
                  <div className="card">
                    <div className="card-header pb-0">
                      <h6>Aspect Sentiment</h6>
                      <p className="text-sm text-muted">
                        Top aspects extracted from comments with sentiment ratios.
                      </p>
                    </div>
                    <div className="card-body">
                      <div className="table-responsive">
                        <table className="table align-items-center mb-0">
                          <thead>
                            <tr>
                              <th>Aspect</th>
                              <th>Mentions</th>
                              <th>Positive</th>
                              <th>Neutral</th>
                              <th>Negative</th>
                            </tr>
                          </thead>
                          <tbody>
                            {aspectSentiment.map((aspect, index) => (
                              <tr key={`${aspect.aspect}-${index}`}>
                                <td>{aspect.aspect}</td>
                                <td>{aspect.count}</td>
                                <td>{formatPercent(aspect.ratio?.Positive)}</td>
                                <td>{formatPercent(aspect.ratio?.Neutral)}</td>
                                <td>{formatPercent(aspect.ratio?.Negative)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Top Comments Section */}
            {topComments && topComments.length > 0 && (
              <div className="row mb-4">
                <div className="col-12">
                  <div className="card">
                    <div className="card-header pb-0">
                      <h6>Most Influential Comments (By Likes)</h6>
                      <p className="text-sm text-muted">These comments received the most engagement from viewers</p>
                    </div>
                    <div className="card-body">
                      <div className="row">
                        {/* Top Positive Comments */}
                        <div className="col-md-6">
                          <h6 className="text-success mb-3">
                            <i className="fas fa-thumbs-up"></i> Top Positive Comments
                          </h6>
                          {topComments
                            .filter((comment) => comment.sentiment === "Positive")
                            .slice(0, 3)
                            .map((comment, index) => (
                              <div
                                key={index}
                                className="card mb-3"
                                style={{
                                  backgroundColor: "#f0f9f0",
                                  border: "1px solid #d4edda",
                                }}
                              >
                                <div className="card-body p-3">
                                  <p className="mb-2" style={{ fontSize: "14px" }}>
                                    "{comment.text}"
                                  </p>
                                  <div className="d-flex justify-content-between align-items-center">
                                    <small className="text-muted">
                                      <i className="fas fa-user"></i> {comment.author}
                                    </small>
                                    <span
                                      className="badge"
                                      style={{
                                        backgroundColor: "#28a745",
                                        color: "white",
                                      }}
                                    >
                                      <i className="fas fa-heart"></i> {comment.likes} likes
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          {topComments.filter((comment) => comment.sentiment === "Positive")
                            .length === 0 && (
                            <p className="text-muted">No highly-liked positive comments</p>
                          )}
                        </div>

                        {/* Top Negative Comments */}
                        <div className="col-md-6">
                          <h6 className="text-danger mb-3">
                            <i className="fas fa-thumbs-down"></i> Top Negative Comments
                          </h6>
                          {topComments
                            .filter((comment) => comment.sentiment === "Negative")
                            .slice(0, 3)
                            .map((comment, index) => (
                              <div
                                key={index}
                                className="card mb-3"
                                style={{
                                  backgroundColor: "#fff0f0",
                                  border: "1px solid #f5c6cb",
                                }}
                              >
                                <div className="card-body p-3">
                                  <p className="mb-2" style={{ fontSize: "14px" }}>
                                    "{comment.text}"
                                  </p>
                                  <div className="d-flex justify-content-between align-items-center">
                                    <small className="text-muted">
                                      <i className="fas fa-user"></i> {comment.author}
                                    </small>
                                    <span
                                      className="badge"
                                      style={{
                                        backgroundColor: "#dc3545",
                                        color: "white",
                                      }}
                                    >
                                      <i className="fas fa-heart"></i> {comment.likes} likes
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          {topComments.filter((comment) => comment.sentiment === "Negative")
                            .length === 0 && (
                            <p className="text-muted">No highly-liked negative comments</p>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Word Clouds Section */}
            {(topWordsPositive.length > 0 || topWordsNegative.length > 0) && (
              <div className="row mb-4">
                <div className="col-12">
                  <div className="card">
                    <div className="card-header pb-0">
                      <h6>Most Frequent Words in Comments</h6>
                      <p className="text-sm text-muted">
                        Common words help identify key themes and topics in viewer feedback
                      </p>
                    </div>
                    <div className="card-body">
                      <div className="row">
                        {/* Positive Words */}
                        {topWordsPositive.length > 0 && (
                          <div className="col-md-6">
                            <h6 className="text-success mb-3">
                              <i className="fas fa-smile"></i> Positive Comment Keywords
                            </h6>
                            <div
                              style={{
                                backgroundColor: "#f8fff8",
                                padding: "20px",
                                borderRadius: "8px",
                                border: "1px solid #d4edda",
                                minHeight: "200px",
                              }}
                            >
                              {topWordsPositive.slice(0, 20).map((item, index) => {
                                const fontSize = 12 + (item.count / topWordsPositive[0]?.count) * 20;
                                return (
                                  <span
                                    key={index}
                                    style={{
                                      fontSize: `${fontSize}px`,
                                      color: "#28a745",
                                      fontWeight: "500",
                                      margin: "5px",
                                      display: "inline-block",
                                      padding: "2px 6px",
                                    }}
                                    title={`${item.word}: ${item.count} mentions`}
                                  >
                                    {item.word}
                                  </span>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Negative Words */}
                        {topWordsNegative.length > 0 && (
                          <div className="col-md-6">
                            <h6 className="text-danger mb-3">
                              <i className="fas fa-frown"></i> Negative Comment Keywords
                            </h6>
                            <div
                              style={{
                                backgroundColor: "#fff8f8",
                                padding: "20px",
                                borderRadius: "8px",
                                border: "1px solid #f5c6cb",
                                minHeight: "200px",
                              }}
                            >
                              {topWordsNegative.slice(0, 20).map((item, index) => {
                                const fontSize = 12 + (item.count / topWordsNegative[0]?.count) * 20;
                                return (
                                  <span
                                    key={index}
                                    style={{
                                      fontSize: `${fontSize}px`,
                                      color: "#dc3545",
                                      fontWeight: "500",
                                      margin: "5px",
                                      display: "inline-block",
                                      padding: "2px 6px",
                                    }}
                                    title={`${item.word}: ${item.count} mentions`}
                                  >
                                    {item.word}
                                  </span>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Filter Statistics Section */}
            {filterStats && (
              <div className="row mb-4">
                <div className="col-12">
                  <div className="card">
                    <div className="card-header pb-0">
                      <h6>Data Quality & Filtering Statistics</h6>
                      <p className="text-sm text-muted">
                        Transparency in how comments were processed and filtered
                      </p>
                    </div>
                    <div className="card-body">
                      <div className="row">
                        <div className="col-md-3">
                          <div className="text-center p-3">
                            <i
                              className="fas fa-filter"
                              style={{ fontSize: "24px", color: "#6c757d" }}
                            ></i>
                            <h4 className="mt-2">{filterStats.total || 0}</h4>
                            <p className="text-sm text-muted mb-0">Total Filtered</p>
                          </div>
                        </div>
                        <div className="col-md-3">
                          <div className="text-center p-3">
                            <i
                              className="fas fa-shield-alt"
                              style={{ fontSize: "24px", color: "#ffc107" }}
                            ></i>
                            <h4 className="mt-2">{filterStats.spam || 0}</h4>
                            <p className="text-sm text-muted mb-0">Spam Removed</p>
                          </div>
                        </div>
                        <div className="col-md-3">
                          <div className="text-center p-3">
                            <i
                              className="fas fa-language"
                              style={{ fontSize: "24px", color: "#17a2b8" }}
                            ></i>
                            <h4 className="mt-2">{filterStats.language || 0}</h4>
                            <p className="text-sm text-muted mb-0">Non-English</p>
                          </div>
                        </div>
                        <div className="col-md-3">
                          <div className="text-center p-3">
                            <i
                              className="fas fa-text-height"
                              style={{ fontSize: "24px", color: "#dc3545" }}
                            ></i>
                            <h4 className="mt-2">{filterStats.short || 0}</h4>
                            <p className="text-sm text-muted mb-0">Too Short</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            
          </div>
        )}
      </main>
      <Fixedplugins />
    </>
  );
}

export default Dashboard;
