import React, { useContext, useState } from "react";
import { useNavigate, Link, NavLink } from "react-router-dom";
import axiosInstance from "../../axios";
import { HashLink } from "react-router-hash-link";
import AuthContext from "../../context/AuthContext";
// import Navbar from "../../Components/Navbar";

function Search() {
  function logoutHandler() {
    logoutUser();
  }

  const navigate = useNavigate();
  const [hasError, setHasError] = useState(false);
  const { logoutUser, isAuthenticated } = useContext(AuthContext);
  const [video_url, setVideoUrl] = useState("");
  const [max_comments, setMaxComments] = useState(200);
  const [useApi, setUseApi] = useState(true);
  const [sentimentModel, setSentimentModel] = useState("meta_learner");
  const [showResearchOptions, setShowResearchOptions] = useState(false);
  const [ensembleModels, setEnsembleModels] = useState(["logreg", "svm", "tfidf"]);
  const [ensembleWeightsOptimization, setEnsembleWeightsOptimization] = useState("pso");
  const [ensembleWeights, setEnsembleWeights] = useState("");
  const [metaLearnerModels, setMetaLearnerModels] = useState(["logreg", "svm", "tfidf"]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6);
  const [bootstrapSamples, setBootstrapSamples] = useState(500);
  const [randomSeed, setRandomSeed] = useState(42);
  const [aspectTopN, setAspectTopN] = useState(12);
  const [aspectMinFreq, setAspectMinFreq] = useState(3);
  const [fuzzyModels, setFuzzyModels] = useState(["logreg", "svm"]);
  const [fuzzyMfType, setFuzzyMfType] = useState("trapezoidal");
  const [fuzzyDefuzzMethod, setFuzzyDefuzzMethod] = useState("bisector");
  const [fuzzyTNorm, setFuzzyTNorm] = useState("product");
  const [fuzzyTConorm, setFuzzyTConorm] = useState("prob_sum");
  const [fuzzyAlphaCut, setFuzzyAlphaCut] = useState(0.1);
  const [fuzzyResolution, setFuzzyResolution] = useState(100);
  const [ensembleWeightsError, setEnsembleWeightsError] = useState("");
  const [modelComparison, setModelComparison] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [searchError, setSearchError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const resolveApiErrorMessage = (status, data) => {
    const apiMessage = data?.msg || data?.message;
    if (apiMessage) {
      return apiMessage;
    }
    if (status === 500) {
      // A bare 500 with no API-supplied message (no `apiMessage` above) can
      // come from our own Django error handler, but also from the Vite dev
      // proxy returning its own 500 page when the backend is unreachable —
      // the browser can't tell those apart, so avoid implying a URL problem.
      return "Server error. The analysis could not be completed — please try again in a moment.";
    }
    if (status === 404) {
      return "Video not found or unavailable.";
    }
    if (status === 401) {
      return "Authentication failed. Please login again.";
    }
    if (status === 429) {
      return "Rate limit exceeded. Please try again later.";
    }
    return "Error analyzing video. Please try again.";
  };

  const parseModelComparison = () => {
    if (!modelComparison) {
      return { value: null, error: null };
    }
    try {
      return { value: JSON.parse(modelComparison), error: null };
    } catch (err) {
      return { value: null, error: "Model comparison JSON must be valid JSON." };
    }
  };

  const parseEnsembleWeights = (rawValue) => {
    if (!rawValue) {
      return null;
    }
    try {
      const parsed = JSON.parse(rawValue);
      const weights = parsed?.weights && typeof parsed.weights === "object" ? parsed.weights : parsed;
      if (!weights || typeof weights !== "object") {
        throw new Error("Invalid weights format.");
      }
      return weights;
    } catch (err) {
      return null;
    }
  };

  const handleEnsembleWeightsFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const raw = reader.result;
        const parsed = JSON.parse(raw);
        const weights = parsed?.weights && typeof parsed.weights === "object" ? parsed.weights : parsed;
        if (!weights || typeof weights !== "object") {
          throw new Error("Invalid weights file.");
        }
        setEnsembleWeights(JSON.stringify(weights));
        setEnsembleWeightsError("");
      } catch (err) {
        setEnsembleWeightsError("Invalid JSON weights file.");
      }
    };
    reader.readAsText(file);
  };

  // YouTube URL validation helper
  const isValidYouTubeUrl = (url) => {
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|embed\/|v\/)|youtu\.be\/)[\w-]+/;
    return youtubeRegex.test(url);
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // youtube/analyze/ runs in the background (app/models.py::AnalysisJob) and
  // returns a job id immediately rather than blocking the request on
  // possibly minutes of fetch+preprocessing+inference — poll for the result
  // instead of awaiting one long HTTP call. Bounded to ~6 minutes so a stuck
  // job fails visibly instead of polling forever.
  const POLL_INTERVAL_MS = 2000;
  const MAX_POLL_ATTEMPTS = 180;

  const pollAnalysisJob = async (jobId) => {
    for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt += 1) {
      await sleep(POLL_INTERVAL_MS);
      const statusResp = await axiosInstance({
        method: "GET",
        url: `youtube/analyze/status/${jobId}/`,
      });
      if (statusResp.data?.status === "done") {
        return { failed: false, data: statusResp.data };
      }
      if (statusResp.data?.status === "failed") {
        return { failed: true, httpStatus: statusResp.status, data: statusResp.data };
      }
      // "pending" or "running" — keep polling.
    }
    throw new Error("Analysis timed out waiting for a result.");
  };

  const searchHandler = async (e) => {
    e.preventDefault();

    // Reset errors
    setHasError(false);
    setSearchError(false);
    setErrorMessage("");

    // Validate video URL
    if (!video_url) {
      setHasError(true);
      setErrorMessage("YouTube URL is required");
      return;
    }

    if (!isValidYouTubeUrl(video_url)) {
      setHasError(true);
      setErrorMessage("Invalid YouTube URL format. Please enter a valid YouTube video URL.");
      return;
    }

    // Validate max_comments (matches the backend's bound: app/views.py
    // caps max_comments at 2000 via _parse_bounded_int).
    if (max_comments < 1 || max_comments > 2000) {
      setHasError(true);
      setErrorMessage("Max comments must be between 1 and 2000");
      return;
    }

    if (sentimentModel === "ensemble" && ensembleWeights) {
      const parsedWeights = parseEnsembleWeights(ensembleWeights);
      if (!parsedWeights) {
        setHasError(true);
        setErrorMessage("Ensemble weights must be valid JSON.");
        return;
      }
    }

    const { value: parsedModelComparison, error: modelComparisonError } = parseModelComparison();
    if (modelComparisonError) {
      setHasError(true);
      setErrorMessage(modelComparisonError);
      return;
    }

    if (sentimentModel === "fuzzy_ensemble") {
      if (!Array.isArray(fuzzyModels) || fuzzyModels.length === 0) {
        setHasError(true);
        setErrorMessage("Select at least one fuzzy base model.");
        return;
      }
      if (Number.isNaN(fuzzyResolution) || fuzzyResolution < 50 || fuzzyResolution > 500) {
        setHasError(true);
        setErrorMessage("Fuzzy resolution must be between 50 and 500.");
        return;
      }
      if (fuzzyAlphaCut < 0 || fuzzyAlphaCut > 1) {
        setHasError(true);
        setErrorMessage("Fuzzy alpha-cut must be between 0 and 1.");
        return;
      }
      if (confidenceThreshold < 0 || confidenceThreshold > 1) {
        setHasError(true);
        setErrorMessage("Confidence threshold must be between 0 and 1.");
        return;
      }
    }

    try {
      setIsLoading(true);
      const resp = await axiosInstance({
        method: "POST",
        url: "youtube/analyze/",
        timeout: 1000 * 180,
        data: {
          video_url: video_url,
          max_comments: max_comments,
          use_api: useApi,
          sentiment_model: sentimentModel,
          ensemble_models: ensembleModels,
          ensemble_weights: ensembleWeights || null,
          ensemble_weights_optimization: ensembleWeightsOptimization,
          meta_learner_models: metaLearnerModels,
          confidence_threshold: confidenceThreshold,
          bootstrap_samples: bootstrapSamples,
          random_seed: randomSeed,
          aspect_top_n: aspectTopN,
          aspect_min_freq: aspectMinFreq,
          fuzzy_models: fuzzyModels,
          fuzzy_mf_type: fuzzyMfType,
          fuzzy_defuzz_method: fuzzyDefuzzMethod,
          fuzzy_t_norm: fuzzyTNorm,
          fuzzy_t_conorm: fuzzyTConorm,
          fuzzy_alpha_cut: fuzzyAlphaCut,
          fuzzy_resolution: fuzzyResolution,
          model_comparison: parsedModelComparison,
        },
      });

      if (resp.status >= 400) {
        setIsLoading(false);
        setSearchError(true);
        setErrorMessage(resolveApiErrorMessage(resp.status, resp.data));
        return;
      }

      if (resp.status === 202 && resp.data?.job_id) {
        // Background job path (the normal case): keep the loading state up
        // while polling so the UI doesn't look "done" mid-analysis.
        const result = await pollAnalysisJob(resp.data.job_id);
        setIsLoading(false);
        if (result.failed) {
          setSearchError(true);
          setErrorMessage(resolveApiErrorMessage(result.httpStatus, result.data));
          return;
        }
        navigate("/dashboard", { state: result.data });
        return;
      }

      // ANALYSIS_RUN_SYNC=true deployments (or the test environment) return
      // the full result directly with no job to poll.
      setIsLoading(false);
      navigate("/dashboard", {
        state: resp.data,
      });
    } catch (e) {
      setIsLoading(false);
      setSearchError(true);

      if (e.code === 'ECONNABORTED') {
        setErrorMessage("Request timeout. The analysis is taking too long. Please try with fewer comments.");
      } else if (e.response) {
        setErrorMessage(resolveApiErrorMessage(e.response.status, e.response.data));
      } else if (e.request) {
        setErrorMessage("Cannot connect to server. Please check if the backend is running.");
      } else {
        setErrorMessage("An unexpected error occurred. Please try again.");
      }
    }
  };
  return (
    <>
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
            YouTube Sentiment
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
                <Link to="/" className="nav-link" aria-current="page">
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

                  <li
                    className="nav-item"
                    style={{ color: "pointer" }}
                    onClick={logoutHandler}
                  >
                    <div className="nav-link" style={{ cursor: "pointer" }}>
                      
                      <div>
                        <span className="nav-link-text ms-1">Logout</span>
                      </div>
                    </div>
                  </li>
                </>
              )}
            </ul>
            
          </div>
        </div>
      </nav>
      <header className="ex-header">
        <div className="container">
          <div className="row">
            <div className="col-xl-10 offset-xl-1">
              <h1 className="text-center">Analyze YouTube Video</h1>
            </div>
          </div>
        </div>
      </header>
      <div className="container rounded bg-white mt-5 mb-5">
        <div className="row">
          <div className="col-md-3 ">
            
          </div>
          <div className="col-md-5 ">
            <div className="p-3 py-5">
              <div className="d-flex justify-content-between align-items-center mb-3">
                <h4 className="text-right">Video Analysis Settings</h4>
              </div>
              <form>
                {(searchError || hasError) && !isLoading && errorMessage && (
                  <div className="alert alert-danger" role="alert">
                    {errorMessage}
                  </div>
                )}
                <div className="row mt-3">
                  <div className="col-md-12">
                    <label className="labels" htmlFor="video-url">YouTube Video URL</label>
                    <input
                      id="video-url"
                      name="video_url"
                      type="text"
                      className="form-control"
                      placeholder="https://www.youtube.com/watch?v=..."
                      value={video_url}
                      onChange={(e) => {
                        setVideoUrl(e.target.value);
                      }}
                      required
                    />
                  </div>
                  <div className="col-md-12 mt-3">
                    <label className="labels" htmlFor="max-comments">Max Comments (1-2000)</label>
                    <input
                      id="max-comments"
                      name="max_comments"
                      type="number"
                      className="form-control"
                      placeholder="200"
                      min="1"
                      max="2000"
                      value={max_comments}
                      onChange={(e) => {
                        const parsed = parseInt(e.target.value, 10);
                        setMaxComments(Number.isNaN(parsed) ? 200 : parsed);
                      }}
                    />
                  </div>
                  <div className="col-md-12 mt-3">
                    <label className="labels">
                      <input
                        type="checkbox"
                        checked={useApi}
                        onChange={(e) => setUseApi(e.target.checked)}
                        style={{ marginRight: "8px" }}
                      />
                      Use YouTube API
                    </label>
                    <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                      Uses the official YouTube Data API (requires `YOUTUBE_API_KEY`). Turn off to use scraper mode.
                    </p>
                  </div>
                  <div className="col-md-12 mt-3">
                    <label className="labels" htmlFor="sentiment-model">Sentiment Model</label>
                    <select
                      id="sentiment-model"
                      className="form-control"
                      value={sentimentModel}
                      onChange={(e) => setSentimentModel(e.target.value)}
                    >
                      <optgroup label="Recommended">
                        <option value="meta_learner">Meta-Learner (stacking) — best F1</option>
                      </optgroup>
                      <optgroup label="Research (Computational Intelligence)">
                        <option value="ensemble">Ensemble NSGA-II — best calibrated</option>
                        <option value="fuzzy_ensemble">Fuzzy Ensemble (uncertainty-aware)</option>
                        <option value="hybrid_dl">Hybrid CNN-BiLSTM (deep learning)</option>
                        <option value="deberta_v3">DeBERTa-v3 (transformer — CPU, experimental)</option>
                      </optgroup>
                      <optgroup label="Baselines">
                        <option value="logreg">Logistic Regression</option>
                        <option value="svm">Linear SVM</option>
                        <option value="tfidf">TF-IDF</option>
                      </optgroup>
                    </select>
                    <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                      Meta-Learner is the recommended default (best macro-F1). Research models showcase the computational-intelligence ensemble methods; baselines are provided for comparison.
                    </p>
                  </div>
                  <div className="col-md-12 mt-3">
                    <label className="labels">
                      <input
                        type="checkbox"
                        checked={showResearchOptions}
                        onChange={(e) => setShowResearchOptions(e.target.checked)}
                        style={{ marginRight: "8px" }}
                      />
                      Show Research Options
                    </label>
                    <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                      Enables CI experiment settings (ensemble weights, fuzzy params, bootstrap, aspects).
                    </p>
                  </div>
                  {showResearchOptions && (
                    <>
                      {sentimentModel === "ensemble" && (
                        <>
                          <div className="col-md-12 mt-3">
                            <label className="labels">Ensemble Base Models</label>
                            <div className="d-flex flex-wrap gap-3">
                              {["logreg", "svm", "tfidf"].map((model) => (
                                <label key={model} className="labels" style={{ marginRight: "12px" }}>
                                  <input
                                    type="checkbox"
                                    checked={ensembleModels.includes(model)}
                                    onChange={(e) => {
                                      if (e.target.checked) {
                                        setEnsembleModels([...ensembleModels, model]);
                                      } else {
                                        setEnsembleModels(ensembleModels.filter((item) => item !== model));
                                      }
                                    }}
                                    style={{ marginRight: "6px" }}
                                  />
                                  {model.toUpperCase()}
                                </label>
                              ))}
                            </div>
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              Select models to combine. Leave empty to use defaults.
                            </p>
                          </div>
                          <div className="col-md-12 mt-3">
                            <label className="labels" htmlFor="weights-optimization">Weight Optimization</label>
                            <select
                              id="weights-optimization"
                              className="form-control"
                              value={ensembleWeightsOptimization}
                              onChange={(e) => setEnsembleWeightsOptimization(e.target.value)}
                            >
                              <option value="pso">PSO — Best F1 (logreg:0.31, svm:0.69, tfidf:0.0)</option>
                              <option value="nsga2">NSGA-II — Best Calibration (logreg:0.92, svm:0.003, tfidf:0.09)</option>
                            </select>
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              Auto-loaded research weights. Overridden if you provide custom JSON below.
                            </p>
                          </div>
                          <div className="col-md-12 mt-3">
                            <label className="labels" htmlFor="ensemble-weights">Custom Ensemble Weights (JSON, optional)</label>
                            <textarea
                              id="ensemble-weights"
                              className="form-control"
                              rows="3"
                              placeholder='{"logreg": 0.4, "svm": 0.4, "tfidf": 0.2}'
                              value={ensembleWeights}
                              onChange={(e) => {
                                setEnsembleWeights(e.target.value);
                                if (ensembleWeightsError) {
                                  setEnsembleWeightsError("");
                                }
                              }}
                            />
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              Optional. Provide JSON weights or upload a JSON file. Example: {"{\"logreg\":0.4,\"svm\":0.4,\"tfidf\":0.2}"}.
                            </p>
                            <input
                              type="file"
                              accept=".json,application/json"
                              className="form-control mt-2"
                              onChange={handleEnsembleWeightsFile}
                            />
                            {ensembleWeightsError && (
                              <p style={{ fontSize: "12px", color: "#b00020", marginTop: "6px" }}>
                                {ensembleWeightsError}
                              </p>
                            )}
                          </div>
                        </>
                      )}
                      {sentimentModel === "meta_learner" && (
                        <>
                          <div className="col-md-12 mt-3">
                            <label className="labels">Meta-Learner Artifact</label>
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              The API uses the server-configured stacking artifact. Client-side path overrides are disabled.
                            </p>
                          </div>
                          <div className="col-md-12 mt-3">
                            <label className="labels">Meta-Learner Base Models</label>
                            <div className="d-flex flex-wrap gap-3">
                              {["logreg", "svm", "tfidf"].map((model) => (
                                <label key={model} className="labels" style={{ marginRight: "12px" }}>
                                  <input
                                    type="checkbox"
                                    checked={metaLearnerModels.includes(model)}
                                    onChange={(e) => {
                                      if (e.target.checked) {
                                        setMetaLearnerModels([...metaLearnerModels, model]);
                                      } else {
                                        setMetaLearnerModels(metaLearnerModels.filter((item) => item !== model));
                                      }
                                    }}
                                    style={{ marginRight: "6px" }}
                                  />
                                  {model.toUpperCase()}
                                </label>
                              ))}
                            </div>
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              Must match the base models used during meta-learner training.
                            </p>
                          </div>
                        </>
                      )}
                      {sentimentModel === "fuzzy_ensemble" && (
                        <>
                          <div className="col-md-12 mt-4">
                            <h6 className="mb-2">Fuzzy Configuration</h6>
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              Adjust fuzzification and inference settings for uncertainty-aware analysis.
                            </p>
                          </div>
                          <div className="col-md-12 mt-2">
                            <label className="labels">Fuzzy Base Models</label>
                            <div className="d-flex flex-wrap gap-3">
                              {["logreg", "svm", "tfidf"].map((model) => (
                                <label key={model} className="labels" style={{ marginRight: "12px" }}>
                                  <input
                                    type="checkbox"
                                    checked={fuzzyModels.includes(model)}
                                    onChange={(e) => {
                                      if (e.target.checked) {
                                        setFuzzyModels([...fuzzyModels, model]);
                                      } else {
                                        setFuzzyModels(fuzzyModels.filter((item) => item !== model));
                                      }
                                    }}
                                    style={{ marginRight: "6px" }}
                                  />
                                  {model.toUpperCase()}
                                </label>
                              ))}
                            </div>
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              Keep this small (1-2 models) for faster inference.
                            </p>
                          </div>
                          <div className="col-md-4 mt-3">
                            <label className="labels" htmlFor="fuzzy-mf-type">MF Type</label>
                            <select
                              id="fuzzy-mf-type"
                              className="form-control"
                              value={fuzzyMfType}
                              onChange={(e) => setFuzzyMfType(e.target.value)}
                            >
                              <option value="triangular">Triangular</option>
                              <option value="trapezoidal">Trapezoidal</option>
                              <option value="gaussian">Gaussian</option>
                            </select>
                          </div>
                          <div className="col-md-4 mt-3">
                            <label className="labels" htmlFor="fuzzy-defuzz-method">Defuzz Method</label>
                            <select
                              id="fuzzy-defuzz-method"
                              className="form-control"
                              value={fuzzyDefuzzMethod}
                              onChange={(e) => setFuzzyDefuzzMethod(e.target.value)}
                            >
                              <option value="centroid">Centroid</option>
                              <option value="bisector">Bisector</option>
                              <option value="mom">MOM</option>
                              <option value="som">SOM</option>
                              <option value="lom">LOM</option>
                              <option value="weighted_average">Weighted Average</option>
                            </select>
                          </div>
                          <div className="col-md-4 mt-3">
                            <label className="labels" htmlFor="fuzzy-resolution">Resolution</label>
                            <input
                              id="fuzzy-resolution"
                              className="form-control"
                              type="number"
                              min="50"
                              max="500"
                              value={fuzzyResolution}
                              onChange={(e) => setFuzzyResolution(parseInt(e.target.value, 10))}
                            />
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              Higher resolution improves stability but increases runtime.
                            </p>
                          </div>
                          <div className="col-md-4 mt-3">
                            <label className="labels" htmlFor="fuzzy-t-norm">T-Norm</label>
                            <select
                              id="fuzzy-t-norm"
                              className="form-control"
                              value={fuzzyTNorm}
                              onChange={(e) => setFuzzyTNorm(e.target.value)}
                            >
                              <option value="min">Min</option>
                              <option value="product">Product</option>
                              <option value="lukasiewicz">Lukasiewicz</option>
                            </select>
                          </div>
                          <div className="col-md-4 mt-3">
                            <label className="labels" htmlFor="fuzzy-t-conorm">T-Conorm</label>
                            <select
                              id="fuzzy-t-conorm"
                              className="form-control"
                              value={fuzzyTConorm}
                              onChange={(e) => setFuzzyTConorm(e.target.value)}
                            >
                              <option value="max">Max</option>
                              <option value="prob_sum">Probabilistic Sum</option>
                              <option value="bounded_sum">Bounded Sum</option>
                            </select>
                          </div>
                          <div className="col-md-4 mt-3">
                            <label className="labels" htmlFor="fuzzy-alpha-cut">Alpha Cut</label>
                            <input
                              id="fuzzy-alpha-cut"
                              className="form-control"
                              type="number"
                              min="0"
                              max="1"
                              step="0.01"
                              value={fuzzyAlphaCut}
                              onChange={(e) => setFuzzyAlphaCut(parseFloat(e.target.value))}
                            />
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              Use 0.0 for no alpha-cut (default).
                            </p>
                          </div>
                        </>
                      )}
                      <div className="col-md-12 mt-3">
                        <label className="labels" htmlFor="model-comparison">Model Comparison (JSON)</label>
                        <textarea
                          id="model-comparison"
                          className="form-control"
                          rows="3"
                          placeholder='[{\"name\":\"LOGREG\",\"accuracy\":0.6884,\"macro_f1\":0.6894}]'
                          value={modelComparison}
                          onChange={(e) => setModelComparison(e.target.value)}
                        />
                        <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                          Optional. Used to render comparison tables in reports.
                        </p>
                      </div>
                      <div className="col-md-4 mt-3">
                        <label className="labels" htmlFor="confidence-threshold">Confidence Threshold</label>
                        <input
                          id="confidence-threshold"
                          className="form-control"
                          type="number"
                          min="0"
                          max="1"
                          step="0.01"
                          value={confidenceThreshold}
                          onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="col-md-4 mt-3">
                        <label className="labels" htmlFor="bootstrap-samples">Bootstrap Samples</label>
                        <input
                          id="bootstrap-samples"
                          className="form-control"
                          type="number"
                          min="100"
                          step="50"
                          value={bootstrapSamples}
                          onChange={(e) => setBootstrapSamples(parseInt(e.target.value, 10))}
                        />
                      </div>
                      <div className="col-md-4 mt-3">
                        <label className="labels" htmlFor="random-seed">Random Seed</label>
                        <input
                          id="random-seed"
                          className="form-control"
                          type="number"
                          min="1"
                          value={randomSeed}
                          onChange={(e) => setRandomSeed(parseInt(e.target.value, 10))}
                        />
                      </div>
                      <div className="col-md-6 mt-3">
                        <label className="labels" htmlFor="aspect-top-n">Aspect Top-N</label>
                        <input
                          id="aspect-top-n"
                          className="form-control"
                          type="number"
                          min="3"
                          value={aspectTopN}
                          onChange={(e) => setAspectTopN(parseInt(e.target.value, 10))}
                        />
                      </div>
                      <div className="col-md-6 mt-3">
                        <label className="labels" htmlFor="aspect-min-freq">Aspect Min Frequency</label>
                        <input
                          id="aspect-min-freq"
                          className="form-control"
                          type="number"
                          min="1"
                          value={aspectMinFreq}
                          onChange={(e) => setAspectMinFreq(parseInt(e.target.value, 10))}
                        />
                      </div>
                    </>
                  )}
                </div>

                <div className="mt-5 text-center">
                  <input
                    className="p-2 mb-2 bg-primary text-white w-45 my-4 mb-2"
                    // className="btn btn-primary profile-button"
                    type="button"
                    onClick={searchHandler}
                    value={isLoading ? `Analyzing...` : `Analyze Video`}
                    disabled={isLoading ? true : false}
                  ></input>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default Search;
