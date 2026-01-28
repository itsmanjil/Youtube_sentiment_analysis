import React, { useContext, useState } from "react";
import { useNavigate, Link, NavLink } from "react-router-dom";
import axiosInstance from "../../axios";
import { HashLink } from "react-router-hash-link";
import axios from "axios";
import AuthContext from "../../context/AuthContext";
// import Navbar from "../../Components/Navbar";

function Search() {
  function logoutHandler() {
    // console.log("logout");
    logoutUser();
  }

  const token = localStorage.getItem("authToken");
  console.log("token", token);
  const navigate = useNavigate();
  const [hasError, setHasError] = useState(false);
  const { logoutUser, authToken } = useContext(AuthContext);
  const [video_url, setVideoUrl] = useState("");
  const [max_comments, setMaxComments] = useState(200);
  const [use_api, setUseApi] = useState(false);
  const [sentimentModel, setSentimentModel] = useState("logreg");
  const [showResearchOptions, setShowResearchOptions] = useState(false);
  const [ensembleModels, setEnsembleModels] = useState(["logreg", "svm", "tfidf"]);
  const [ensembleWeights, setEnsembleWeights] = useState("");
  const [metaLearnerPath, setMetaLearnerPath] = useState("");
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
      return "Server error. Please check the URL and try again.";
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
      return null;
    }
    try {
      return JSON.parse(modelComparison);
    } catch (err) {
      console.warn("Invalid model comparison JSON:", err);
      return null;
    }
  };

  // YouTube URL validation helper
  const isValidYouTubeUrl = (url) => {
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|embed\/|v\/)|youtu\.be\/)[\w-]+/;
    return youtubeRegex.test(url);
  };

  const searchHandler = async (e) => {
    e.preventDefault();
    console.log("Analyze button clicked");

    // Reset errors
    setHasError(false);
    setSearchError(false);
    setErrorMessage("");

    // Validate video URL
    if (!video_url) {
      console.log("Empty video URL");
      setHasError(true);
      setErrorMessage("YouTube URL is required");
      return;
    }

    if (!isValidYouTubeUrl(video_url)) {
      setHasError(true);
      setErrorMessage("Invalid YouTube URL format. Please enter a valid YouTube video URL.");
      return;
    }

    // Validate max_comments
    if (max_comments < 1 || max_comments > 1000) {
      setHasError(true);
      setErrorMessage("Max comments must be between 1 and 1000");
      return;
    }

    try {
      setIsLoading(true);
      const resp = await axios({
        method: "POST",
        url: `http://127.0.0.1:8000/api/youtube/analyze/`,
        timeout: 1000 * 180,
        validateStatus: (status) => {
          return status < 500;
        },
        data: {
          video_url: video_url,
          max_comments: max_comments,
          use_api: use_api,
          sentiment_model: sentimentModel,
          ensemble_models: ensembleModels,
          ensemble_weights: ensembleWeights || null,
          meta_learner_path: metaLearnerPath || null,
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
          model_comparison: parseModelComparison(),
        },
        headers: {
          Authorization: authToken
            ? "Bearer " + String(authToken.access)
            : null,
          "Content-Type": "application/json",
          accept: "application/json",
        },
      });
      console.log("YouTube analysis response:", resp.data);
      setIsLoading(false);
      if (resp.status >= 400) {
        setSearchError(true);
        setErrorMessage(resolveApiErrorMessage(resp.status, resp.data));
        return;
      }
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

      console.error("Analysis error:", e);
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
                    <label className="labels" htmlFor="max-comments">Max Comments (1-1000)</label>
                    <input
                      id="max-comments"
                      name="max_comments"
                      type="number"
                      className="form-control"
                      placeholder="200"
                      min="1"
                      max="1000"
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
                        checked={use_api}
                        onChange={(e) => setUseApi(e.target.checked)}
                        style={{ marginRight: "8px" }}
                      />
                      Use YouTube API (faster, requires API key)
                    </label>
                    <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                      Uncheck to use scraper mode (slower but no API key needed)
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
                      <option value="logreg">Logistic Regression (trained)</option>
                      <option value="svm">Linear SVM (trained)</option>
                      <option value="tfidf">TF-IDF (legacy)</option>
                      <option value="ensemble">Ensemble (custom weights)</option>
                      <option value="meta_learner">Meta-Learner (stacking)</option>
                      <option value="hybrid_dl">Hybrid CNN-BiLSTM-Attention</option>
                      <option value="bert">Transformer (BERT)</option>
                    </select>
                    <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                      Choose a model for inference. Enable research options to customize CI settings and export metadata.
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
                            <label className="labels" htmlFor="ensemble-weights">Ensemble Weights (JSON)</label>
                            <textarea
                              id="ensemble-weights"
                              className="form-control"
                              rows="3"
                              placeholder='{"logreg": 0.4, "svm": 0.4, "tfidf": 0.2}'
                              value={ensembleWeights}
                              onChange={(e) => setEnsembleWeights(e.target.value)}
                            />
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              Optional. Provide JSON weights or a file path. Example: {"{\"logreg\":0.4,\"svm\":0.4,\"tfidf\":0.2}"}.
                            </p>
                          </div>
                        </>
                      )}
                      {sentimentModel === "meta_learner" && (
                        <>
                          <div className="col-md-12 mt-3">
                            <label className="labels" htmlFor="meta-learner-path">Meta-Learner Model Path</label>
                            <input
                              id="meta-learner-path"
                              className="form-control"
                              type="text"
                              placeholder="backend/models/meta_learner.pkl"
                              value={metaLearnerPath}
                              onChange={(e) => setMetaLearnerPath(e.target.value)}
                            />
                            <p style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>
                              Path to stacking model artifact (e.g., backend/models/meta_learner.pkl).
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
