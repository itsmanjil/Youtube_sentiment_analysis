import React, { useState, useEffect } from "react";
import "./report.css";
import { useLocation, Link, useNavigate } from "react-router-dom";

export default function Report() {
  const location = useLocation();
  const navigate = useNavigate();

  const sentimentData = location.state;
  const user_name = sentimentData?.user_name || "Unknown User";

  const [totalComments, setTotalComments] = useState(0);
  const confidenceStats = sentimentData?.confidenceStats || null;
  const uncertaintyStats = sentimentData?.uncertaintyStats || null;
  const confidenceIntervals = sentimentData?.confidenceIntervals || null;
  const aspectSentiment = sentimentData?.aspectSentiment || [];
  const analysisMeta = sentimentData?.analysisMeta || null;
  const modelUsed = sentimentData?.modelUsed || null;
  const sentimentTimeline = sentimentData?.sentimentTimeline || [];
  const ensembleModels = analysisMeta?.ensemble?.models || [];
  const ensembleWeights = analysisMeta?.ensemble?.weights || null;
  const ensembleWeightsSource = analysisMeta?.ensemble?.weights_source || null;
  const metaLearnerInfo = analysisMeta?.meta_learner || null;
  const fuzzyInfo = analysisMeta?.fuzzy || null;
  const calibrationInfo =
    analysisMeta?.calibration ||
    analysisMeta?.calibration_metrics ||
    analysisMeta?.fuzzy?.calibration ||
    analysisMeta?.fuzzy?.calibration_metrics ||
    null;
  const calibrationECE =
    calibrationInfo?.ece ??
    calibrationInfo?.expected_calibration_error ??
    null;
  const calibrationBrier =
    calibrationInfo?.brier ??
    calibrationInfo?.brier_score ??
    null;
  const calibrationMCE =
    calibrationInfo?.mce ??
    calibrationInfo?.maximum_calibration_error ??
    null;
  const calibrationBins = calibrationInfo?.bins ?? null;
  const calibrationTemperature = calibrationInfo?.temperature ?? null;
  const calibrationApplied = calibrationInfo?.applied ?? false;
  const hasCalibration =
    calibrationECE !== null ||
    calibrationBrier !== null ||
    calibrationMCE !== null ||
    calibrationTemperature !== null;
  const ensembleWeightSummary = ensembleWeights
    ? Object.entries(ensembleWeights)
        .map(([key, value]) => `${key}: ${value}`)
        .join(", ")
    : null;

  const formatPercent = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return "0%";
    }
    return `${(value * 100).toFixed(1)}%`;
  };

  const exportModelComparisonCSV = () => {
    if (!Array.isArray(analysisMeta?.model_comparison) || analysisMeta.model_comparison.length === 0) {
      return;
    }
    const headers = ["Model", "Accuracy", "Macro F1"];
    const rows = analysisMeta.model_comparison.map((row) => [
      row.name || row.model || "",
      row.accuracy ?? "",
      row.macro_f1 ?? "",
    ]);
    const csv = [headers, ...rows]
      .map((r) => r.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(","))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "model_comparison.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const sentimentTotals = (sentimentData?.sentimentBreakdown || []).reduce(
    (acc, item) => {
      const label = item.sentiment || item.name || "";
      if (label) {
        acc[label] = item.value;
      }
      return acc;
    },
    {}
  );
  const handlePrint = () => {
    window.print();
  };

  useEffect(() => {
    if (!sentimentData) {
      navigate("/dashboard");
    }
  }, [sentimentData, navigate]);

  useEffect(() => {
    if (!sentimentData) {
      setTotalComments(0);
      return;
    }
    let total = 0;
    (sentimentData?.sentimentBreakdown || []).forEach((sentiment) => {
      total += sentiment.value;
    });

    setTotalComments(total);
  }, [sentimentData]);

  // Validate that location.state exists
  if (!sentimentData) {
    return (
      <div className="container mt-5">
        <div className="alert alert-warning" role="alert">
          No report data available. Redirecting to dashboard...
        </div>
      </div>
    );
  }

  return (
    <div className="container-main">
      <div className="container-nav-wrapper d-print-none">
        <div className="dropdown float-rg-end pe-4 d-print-none">
          <Link
            to="/dashboard"
            style={{ textDecoration: "none", color: "black" }}
          >
            <span className="fa fa-arrow-left text-dark" style={{ fontSize: '28px' }}></span>
          </Link>
        </div>

        <div className="dropdown float-lg-end pe-4 d-print-none">
          <button
            onClick={handlePrint}
            className="btn btn-dark"
            // style={{ border: "none", background: "transparent" }}
          >
            <span className="fas fa-print"></span> Print
          </button>
        </div>
      </div>

      <div className="my-5 page" size="A4">
        <div id="pagePrint">
          <div className="p-5" id="printPage">
            <section className="top-content bb d-flex justify-content-between">
              <div className="logo">
                <h2>YouTube Sentiment Report</h2>
                {/* <!-- <img src="logo.png" alt="" className="img-fluid"> --> */}
              </div>
              <div className="top-left">
                <div className="graphic-path">
                  <p>Report</p>
                </div>
                <div className="position-relative">
                  <p>
                    Report no.:<span>001</span>
                  </p>
                </div>
              </div>
            </section>

            <section className="store-user mt-5">
              <div className="col-10">
                <div className="row bb pb-3">
                  <div className="col-7">
                    <p>Video analyzed:</p>
                    <h2>{sentimentData.videoTitle}</h2>
                    {/* <p className="address"> 777 Brockton Avenue, <br/> Abington MA 2351, <br/>Vestavia Hills AL </p> */}
                  </div>
                  <div className="col-5">
                    <p>Searched By:</p>
                    <h2>{user_name}</h2>
                    {/* <p className="address"> email <br/> Abington MA 2351, <br/>Vestavia Hills AL </p> */}
                  </div>
                </div>
                <div className="row extra-info pt-3">
                  <div className="col-7">
                    <p>
                      Total Comments Analyzed: <span>{totalComments}</span>
                    </p>
                  </div>
                  <div className="col-5">
                    <p>
                      {" "}
                      Date: <span>{sentimentData.fetchedDate}</span>
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <section className="product-area mt-4">
              <table className="table table-hover">
                <thead>
                  <tr>
                    <td>Sentiment</td>
                    <td>Total</td>
                  </tr>
                </thead>
                <tbody>
                  {(sentimentData?.sentimentBreakdown || []).map((sentiment, index) => {
                    const label =
                      sentiment.sentiment || sentiment.name || `item-${index}`;
                    return (
                      <tr key={`${label}-${index}`}>
                        <td>
                          <div className="media">
                            <div className="media-body">
                              <p className="mt-0 title">
                                {label}
                              </p>
                            </div>
                          </div>
                        </td>
                        <td>{sentiment.value}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </section>

            <section className="balance-info">
              <div className="row">
                <div className="col-12">
                  <p className="m-0 font-weight-bold"> Note: </p>
                  <p>
                    Your analysis for {sentimentData.videoTitle} processed {totalComments} comments.
                    It includes {sentimentTotals.Positive || 0} positive,{" "}
                    {sentimentTotals.Negative || 0} negative, and{" "}
                    {sentimentTotals.Neutral || 0} neutral comments.
                  </p>
                </div>
                

                
              </div>
            </section>

            {(modelUsed || analysisMeta) && (
              <section className="product-area mt-4">
                <h6>Model & Experiment Settings</h6>
                <table className="table table-hover">
                  <tbody>
                    <tr>
                      <td>Model Used</td>
                      <td>{modelUsed || "N/A"}</td>
                    </tr>
                    {ensembleModels.length > 0 && (
                      <tr>
                        <td>Ensemble Models</td>
                        <td>{ensembleModels.join(", ")}</td>
                      </tr>
                    )}
                    {ensembleWeightSummary && (
                      <tr>
                        <td>Ensemble Weights</td>
                        <td>{ensembleWeightSummary}</td>
                      </tr>
                    )}
                    {ensembleWeightsSource && (
                      <tr>
                        <td>Ensemble Weights Source</td>
                        <td>{ensembleWeightsSource}</td>
                      </tr>
                    )}
                    {metaLearnerInfo && (
                      <>
                        {metaLearnerInfo.model_artifact && (
                          <tr>
                            <td>Meta-Learner Artifact</td>
                            <td>{metaLearnerInfo.model_artifact}</td>
                          </tr>
                        )}
                        {metaLearnerInfo.base_models && (
                          <tr>
                            <td>Meta-Learner Base Models</td>
                            <td>{metaLearnerInfo.base_models.join(", ")}</td>
                          </tr>
                        )}
                        {metaLearnerInfo.feature_type && (
                          <tr>
                            <td>Meta-Learner Feature Type</td>
                            <td>{metaLearnerInfo.feature_type}</td>
                          </tr>
                        )}
                        {metaLearnerInfo.meta_learner_type && (
                          <tr>
                            <td>Meta-Learner Type</td>
                            <td>{metaLearnerInfo.meta_learner_type}</td>
                          </tr>
                        )}
                      </>
                    )}
                    {fuzzyInfo && (
                      <>
                        {fuzzyInfo.base_models && (
                          <tr>
                            <td>Fuzzy Base Models</td>
                            <td>{fuzzyInfo.base_models.join(", ")}</td>
                          </tr>
                        )}
                        <tr>
                          <td>Fuzzy Routing</td>
                          <td>{fuzzyInfo.nf_gate_active ? "Neuro-Fuzzy Gate (ANFIS)" : "Static Fuzzy Inference"}</td>
                        </tr>
                        {fuzzyInfo.mf_type && (
                          <tr>
                            <td>Fuzzy MF Type</td>
                            <td>{fuzzyInfo.mf_type}</td>
                          </tr>
                        )}
                        {fuzzyInfo.defuzz_method && (
                          <tr>
                            <td>Defuzz Method</td>
                            <td>{fuzzyInfo.defuzz_method}</td>
                          </tr>
                        )}
                        {fuzzyInfo.t_norm && (
                          <tr>
                            <td>T-Norm</td>
                            <td>{fuzzyInfo.t_norm}</td>
                          </tr>
                        )}
                        {fuzzyInfo.t_conorm && (
                          <tr>
                            <td>T-Conorm</td>
                            <td>{fuzzyInfo.t_conorm}</td>
                          </tr>
                        )}
                        {fuzzyInfo.alpha_cut !== undefined && (
                          <tr>
                            <td>Alpha Cut</td>
                            <td>{fuzzyInfo.alpha_cut}</td>
                          </tr>
                        )}
                      </>
                    )}
                    {analysisMeta?.bootstrap_samples !== undefined && (
                      <tr>
                        <td>Bootstrap Samples</td>
                        <td>{analysisMeta.bootstrap_samples}</td>
                      </tr>
                    )}
                    {analysisMeta?.random_seed !== undefined && (
                      <tr>
                        <td>Random Seed</td>
                        <td>{analysisMeta.random_seed}</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </section>
            )}
            {hasCalibration && (
              <section className="product-area mt-4">
                <h6>Calibration Metrics</h6>
                <table className="table table-hover">
                  <tbody>
                    {calibrationECE !== null && (
                      <tr>
                        <td>Expected Calibration Error (ECE)</td>
                        <td>{calibrationECE}</td>
                      </tr>
                    )}
                    {calibrationBrier !== null && (
                      <tr>
                        <td>Multiclass Brier Score</td>
                        <td>{calibrationBrier}</td>
                      </tr>
                    )}
                    {calibrationMCE !== null && (
                      <tr>
                        <td>Maximum Calibration Error (MCE)</td>
                        <td>{calibrationMCE}</td>
                      </tr>
                    )}
                    {calibrationBins !== null && (
                      <tr>
                        <td>Calibration Bins</td>
                        <td>{calibrationBins}</td>
                      </tr>
                    )}
                    {calibrationTemperature !== null && (
                      <tr>
                        <td>Temperature Scaling (T)</td>
                        <td>
                          {calibrationTemperature?.toFixed(4)}{" "}
                          {calibrationApplied ? "(applied)" : "(loaded, not applied)"}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </section>
            )}
            {Array.isArray(analysisMeta?.model_comparison) && analysisMeta.model_comparison.length > 0 && (
              <section className="product-area mt-4">
                <div className="d-flex align-items-center justify-content-between">
                  <h6 className="mb-0">Model Comparison (Research Mode)</h6>
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-secondary"
                    onClick={exportModelComparisonCSV}
                  >
                    Export CSV
                  </button>
                </div>
                <table className="table table-hover">
                  <thead>
                    <tr>
                      <td>Model</td>
                      <td>Accuracy</td>
                      <td>Macro F1</td>
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
              </section>
            )}

            {confidenceStats && (
              <section className="product-area mt-4">
                <h6>Confidence Summary</h6>
                <table className="table table-hover">
                  <tbody>
                    <tr>
                      <td>Mean Confidence</td>
                      <td>{formatPercent(confidenceStats.mean)}</td>
                    </tr>
                    <tr>
                      <td>Median Confidence</td>
                      <td>{formatPercent(confidenceStats.median)}</td>
                    </tr>
                    <tr>
                      <td>Low-Confidence Ratio</td>
                      <td>{formatPercent(confidenceStats.low_confidence_ratio)}</td>
                    </tr>
                    {confidenceStats.threshold !== undefined && (
                      <tr>
                        <td>Low-Confidence Threshold</td>
                        <td>{confidenceStats.threshold}</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </section>
            )}

            {uncertaintyStats && (
              <section className="product-area mt-4">
                <h6>Model Uncertainty (Shannon Entropy)</h6>
                <table className="table table-hover">
                  <tbody>
                    <tr>
                      <td>Mean Entropy</td>
                      <td>{(uncertaintyStats.mean_entropy * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                      <td>Min Entropy</td>
                      <td>{(uncertaintyStats.min_entropy * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                      <td>Max Entropy</td>
                      <td>{(uncertaintyStats.max_entropy * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                      <td>High Uncertainty Ratio (&gt;50% entropy)</td>
                      <td>{(uncertaintyStats.high_uncertainty_ratio * 100).toFixed(1)}%</td>
                    </tr>
                  </tbody>
                </table>
              </section>
            )}

            {confidenceIntervals && (
              <section className="product-area mt-4">
                <h6>Sentiment Confidence Intervals (95%)</h6>
                <table className="table table-hover">
                  <thead>
                    <tr>
                      <td>Sentiment</td>
                      <td>Lower</td>
                      <td>Upper</td>
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
              </section>
            )}

            {aspectSentiment.length > 0 && (
              <section className="product-area mt-4">
                <h6>Aspect Sentiment (Top {Math.min(10, aspectSentiment.length)})</h6>
                <table className="table table-hover">
                  <thead>
                    <tr>
                      <td>Aspect</td>
                      <td>Mentions</td>
                      <td>Positive</td>
                      <td>Neutral</td>
                      <td>Negative</td>
                    </tr>
                  </thead>
                  <tbody>
                    {aspectSentiment.slice(0, 10).map((aspect, index) => (
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
              </section>
            )}

            {sentimentTimeline.length > 0 && (
              <section className="product-area mt-4">
                <h6>Sentiment Timeline (Hourly Snapshot)</h6>
                <table className="table table-hover">
                  <thead>
                    <tr>
                      <td>Time</td>
                      <td>Positive</td>
                      <td>Neutral</td>
                      <td>Negative</td>
                    </tr>
                  </thead>
                  <tbody>
                    {sentimentTimeline.slice(0, 12).map((row, index) => (
                      <tr key={`${row.time}-${index}`}>
                        <td>{row.time}</td>
                        <td>{row.positive}</td>
                        <td>{row.neutral}</td>
                        <td>{row.negative}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}

            

            
          </div>
        </div>
      </div>
    </div>
  );
}
