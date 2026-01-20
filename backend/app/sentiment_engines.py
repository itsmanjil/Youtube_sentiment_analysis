
from dataclasses import dataclass
from pathlib import Path
import pickle

from .analysis_utils import normalize_probs, SENTIMENT_LABELS


def normalize_label(label):
    if label is None:
        return "Neutral"
    key = str(label).strip()
    if not key:
        return "Neutral"
    normalized = key.lower()
    label_map = {
        "positive": "Positive",
        "pos": "Positive",
        "negative": "Negative",
        "neg": "Negative",
        "neutral": "Neutral",
        "neu": "Neutral",
        "label_0": "Negative",
        "label_1": "Neutral",
        "label_2": "Positive",
    }
    return label_map.get(normalized, "Neutral")


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float
    probs: dict
    model: str
    raw: dict = None

    def to_dict(self):
        return {
            "label": self.label,
            "score": self.score,
            "probs": self.probs,
            "model": self.model,
            "raw": self.raw,
        }


def coerce_sentiment_result(result, model_name):
    if isinstance(result, SentimentResult):
        return result
    if isinstance(result, dict) and "label" in result:
        label = normalize_label(result.get("label"))
        probs = normalize_probs(result.get("probs", {label: 1.0}))
        score = float(result.get("score", 0.0))
        return SentimentResult(label=label, score=score, probs=probs, model=model_name, raw=result)
    if isinstance(result, tuple) and len(result) == 2:
        label, score = result
        normalized_label = normalize_label(label)
        probs = normalize_probs({normalized_label: 1.0})
        return SentimentResult(
            label=normalized_label,
            score=float(score),
            probs=probs,
            model=model_name,
            raw={"legacy": True, "label": label, "score": score},
        )
    return SentimentResult(
        label="Neutral",
        score=0.0,
        probs=normalize_probs({}),
        model=model_name,
        raw={"unparsed_result": True},
    )


class TFIDFSentimentEngine:

    def __init__(self, model_path='./models/tfidf/model.sav',
                 vectorizer_path='./models/tfidf/tfidfVectorizer.pickle'):
        import pickle

        model_path = _resolve_model_path(model_path)
        vectorizer_path = _resolve_model_path(vectorizer_path)
        try:
            self.model = pickle.load(open(model_path, 'rb'))
            self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model files not found. Expected:\n"
                f"  - {model_path}\n"
                f"  - {vectorizer_path}"
            )
        self._validate_fitted()

    def _validate_fitted(self):
        try:
            import sklearn
            from sklearn.utils.validation import check_is_fitted
        except ImportError:
            return

        errors = []
        try:
            check_is_fitted(self.vectorizer, "vocabulary_")
            if hasattr(self.vectorizer, "_tfidf"):
                check_is_fitted(self.vectorizer._tfidf, "idf_")
        except Exception as exc:
            errors.append(f"Vectorizer not fitted ({exc.__class__.__name__}).")

        try:
            check_is_fitted(self.model, "classes_")
        except Exception as exc:
            errors.append(f"Classifier not fitted ({exc.__class__.__name__}).")

        if errors:
            model_version = getattr(self.model, "__sklearn_version__", None)
            vectorizer_version = getattr(self.vectorizer, "__sklearn_version__", None)
            version_note = ""
            if model_version or vectorizer_version:
                version_note = (
                    f" Model version: {model_version or 'unknown'},"
                    f" vectorizer version: {vectorizer_version or 'unknown'},"
                    f" runtime version: {sklearn.__version__}."
                )
            raise RuntimeError(
                "TF-IDF model/vectorizer is not fitted or is incompatible with "
                f"the current scikit-learn version.{version_note} "
                "Reinstall scikit-learn==0.24.2 or retrain/re-serialize the model "
                "with your current environment."
            )

    def _predict_probs(self, vector):
        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(vector)
            labels = getattr(self.model, "classes_", [])
            if len(getattr(raw_probs, "shape", [])) == 1:
                raw_probs = [raw_probs]
            mapped_rows = []
            for row in raw_probs:
                mapped = {
                    normalize_label(label): float(row[idx])
                    for idx, label in enumerate(labels)
                }
                mapped_rows.append(normalize_probs(mapped))
            return mapped_rows
        return None

    def analyze(self, text):
        import pandas as pd

        df = pd.DataFrame([{'tweet': text}])
        tweet_vec = self.vectorizer.transform(df['tweet'])
        prediction = self.model.predict(tweet_vec)[0]

        sentiment = normalize_label(prediction)
        probs = self._predict_probs(tweet_vec)
        if probs is None:
            probs = normalize_probs({sentiment: 1.0})
        elif isinstance(probs, list):
            probs = probs[0]

        return SentimentResult(
            label=sentiment,
            score=float(probs.get(sentiment, 0.0)),
            probs=probs,
            model="tfidf",
            raw={"prediction": prediction},
        )

    def batch_analyze(self, texts):
        import pandas as pd

        df = pd.DataFrame([{'tweet': text} for text in texts])
        tweet_vec = self.vectorizer.transform(df['tweet'])
        predictions = self.model.predict(tweet_vec)

        probs = self._predict_probs(tweet_vec)
        results = []
        for idx, pred in enumerate(predictions):
            sentiment = normalize_label(pred)
            if probs:
                row_probs = probs[idx] if isinstance(probs, list) else probs
                sentiment_probs = {
                    label: row_probs.get(label, 0.0)
                    for label in SENTIMENT_LABELS
                }
            else:
                sentiment_probs = normalize_probs({sentiment: 1.0})

            results.append(SentimentResult(
                label=sentiment,
                score=float(sentiment_probs.get(sentiment, 0.0)),
                probs=sentiment_probs,
                model="tfidf",
                raw={"prediction": pred},
            ))

        return results


class LogRegSentimentEngine:

    def __init__(self, model_path='./models/logreg/model.sav',
                 vectorizer_path='./models/logreg/tfidfVectorizer.pickle'):
        import pickle

        model_path = _resolve_model_path(model_path)
        vectorizer_path = _resolve_model_path(vectorizer_path)
        try:
            self.model = pickle.load(open(model_path, 'rb'))
            self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model files not found. Expected:\n"
                f"  - {model_path}\n"
                f"  - {vectorizer_path}"
            )
        self._validate_fitted()

    def _validate_fitted(self):
        try:
            import sklearn
            from sklearn.utils.validation import check_is_fitted
        except ImportError:
            return

        errors = []
        try:
            check_is_fitted(self.vectorizer, "vocabulary_")
            if hasattr(self.vectorizer, "_tfidf"):
                check_is_fitted(self.vectorizer._tfidf, "idf_")
        except Exception as exc:
            errors.append(f"Vectorizer not fitted ({exc.__class__.__name__}).")

        try:
            check_is_fitted(self.model, "classes_")
        except Exception as exc:
            errors.append(f"Classifier not fitted ({exc.__class__.__name__}).")

        if errors:
            model_version = getattr(self.model, "__sklearn_version__", None)
            vectorizer_version = getattr(self.vectorizer, "__sklearn_version__", None)
            version_note = ""
            if model_version or vectorizer_version:
                version_note = (
                    f" Model version: {model_version or 'unknown'},"
                    f" vectorizer version: {vectorizer_version or 'unknown'},"
                    f" runtime version: {sklearn.__version__}."
                )
            raise RuntimeError(
                "LogReg model/vectorizer is not fitted or is incompatible with "
                f"the current scikit-learn version.{version_note}"
            )

    def _predict_probs(self, vector):
        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(vector)
            labels = getattr(self.model, "classes_", [])
            if len(getattr(raw_probs, "shape", [])) == 1:
                raw_probs = [raw_probs]
            mapped_rows = []
            for row in raw_probs:
                mapped = {
                    normalize_label(label): float(row[idx])
                    for idx, label in enumerate(labels)
                }
                mapped_rows.append(normalize_probs(mapped))
            return mapped_rows
        return None

    def analyze(self, text):
        import pandas as pd

        df = pd.DataFrame([{'text': text}])
        text_vec = self.vectorizer.transform(df['text'])
        prediction = self.model.predict(text_vec)[0]

        sentiment = normalize_label(prediction)
        probs = self._predict_probs(text_vec)
        if probs is None:
            probs = normalize_probs({sentiment: 1.0})
        elif isinstance(probs, list):
            probs = probs[0]

        return SentimentResult(
            label=sentiment,
            score=float(probs.get(sentiment, 0.0)),
            probs=probs,
            model="logreg",
            raw={"prediction": prediction},
        )

    def batch_analyze(self, texts):
        import pandas as pd

        df = pd.DataFrame([{'text': text} for text in texts])
        text_vec = self.vectorizer.transform(df['text'])
        predictions = self.model.predict(text_vec)

        probs = self._predict_probs(text_vec)
        results = []
        for idx, pred in enumerate(predictions):
            sentiment = normalize_label(pred)
            if probs:
                row_probs = probs[idx] if isinstance(probs, list) else probs
                sentiment_probs = {
                    label: row_probs.get(label, 0.0)
                    for label in SENTIMENT_LABELS
                }
            else:
                sentiment_probs = normalize_probs({sentiment: 1.0})

            results.append(SentimentResult(
                label=sentiment,
                score=float(sentiment_probs.get(sentiment, 0.0)),
                probs=sentiment_probs,
                model="logreg",
                raw={"prediction": pred},
            ))

        return results


class SVMSentimentEngine:

    def __init__(self, model_path='./models/svm/model.sav',
                 vectorizer_path='./models/svm/tfidfVectorizer.pickle'):
        import pickle

        model_path = _resolve_model_path(model_path)
        vectorizer_path = _resolve_model_path(vectorizer_path)
        try:
            self.model = pickle.load(open(model_path, 'rb'))
            self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model files not found. Expected:\n"
                f"  - {model_path}\n"
                f"  - {vectorizer_path}"
            )
        self._validate_fitted()

    def _validate_fitted(self):
        try:
            import sklearn
            from sklearn.utils.validation import check_is_fitted
        except ImportError:
            return

        errors = []
        try:
            check_is_fitted(self.vectorizer, "vocabulary_")
            if hasattr(self.vectorizer, "_tfidf"):
                check_is_fitted(self.vectorizer._tfidf, "idf_")
        except Exception as exc:
            errors.append(f"Vectorizer not fitted ({exc.__class__.__name__}).")

        try:
            check_is_fitted(self.model, "classes_")
        except Exception as exc:
            errors.append(f"Classifier not fitted ({exc.__class__.__name__}).")

        if errors:
            model_version = getattr(self.model, "__sklearn_version__", None)
            vectorizer_version = getattr(self.vectorizer, "__sklearn_version__", None)
            version_note = ""
            if model_version or vectorizer_version:
                version_note = (
                    f" Model version: {model_version or 'unknown'},"
                    f" vectorizer version: {vectorizer_version or 'unknown'},"
                    f" runtime version: {sklearn.__version__}."
                )
            raise RuntimeError(
                "SVM model/vectorizer is not fitted or is incompatible with "
                f"the current scikit-learn version.{version_note}"
            )

    def _predict_probs(self, vector):
        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(vector)
            labels = getattr(self.model, "classes_", [])
            if len(getattr(raw_probs, "shape", [])) == 1:
                raw_probs = [raw_probs]
            mapped_rows = []
            for row in raw_probs:
                mapped = {
                    normalize_label(label): float(row[idx])
                    for idx, label in enumerate(labels)
                }
                mapped_rows.append(normalize_probs(mapped))
            return mapped_rows
        if hasattr(self.model, "decision_function"):
            import numpy as np

            scores = self.model.decision_function(vector)
            if getattr(scores, "ndim", 1) == 1:
                scores = np.vstack([-scores, scores]).T
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            labels = getattr(self.model, "classes_", [])
            mapped_rows = []
            for row in probs:
                mapped = {
                    normalize_label(label): float(row[idx])
                    for idx, label in enumerate(labels)
                }
                mapped_rows.append(normalize_probs(mapped))
            return mapped_rows
        return None

    def analyze(self, text):
        import pandas as pd

        df = pd.DataFrame([{'text': text}])
        text_vec = self.vectorizer.transform(df['text'])
        prediction = self.model.predict(text_vec)[0]

        sentiment = normalize_label(prediction)
        probs = self._predict_probs(text_vec)
        if probs is None:
            probs = normalize_probs({sentiment: 1.0})
        elif isinstance(probs, list):
            probs = probs[0]

        return SentimentResult(
            label=sentiment,
            score=float(probs.get(sentiment, 0.0)),
            probs=probs,
            model="svm",
            raw={"prediction": prediction},
        )

    def batch_analyze(self, texts):
        import pandas as pd

        df = pd.DataFrame([{'text': text} for text in texts])
        text_vec = self.vectorizer.transform(df['text'])
        predictions = self.model.predict(text_vec)

        probs = self._predict_probs(text_vec)
        results = []
        for idx, pred in enumerate(predictions):
            sentiment = normalize_label(pred)
            if probs:
                row_probs = probs[idx] if isinstance(probs, list) else probs
                sentiment_probs = {
                    label: row_probs.get(label, 0.0)
                    for label in SENTIMENT_LABELS
                }
            else:
                sentiment_probs = normalize_probs({sentiment: 1.0})

            results.append(SentimentResult(
                label=sentiment,
                score=float(sentiment_probs.get(sentiment, 0.0)),
                probs=sentiment_probs,
                model="svm",
                raw={"prediction": pred},
            ))

        return results


class EnsembleSentimentEngine:

    def __init__(self, base_models=None, weights=None):
        if base_models is None:
            base_models = ["logreg", "svm", "tfidf"]
        self.requested_models = base_models
        self.engines = {}
        self.model_errors = {}
        for model in base_models:
            try:
                self.engines[model] = get_base_engine(model)
            except Exception as exc:
                self.model_errors[model] = str(exc)
        if not self.engines:
            raise RuntimeError(
                "No ensemble base models could be initialized. "
                f"Errors: {self.model_errors}"
            )
        self.base_models = list(self.engines.keys())
        self.weights = self._normalize_weights(weights)

    def _normalize_weights(self, weights):
        if weights is None:
            default_weights = {
                "logreg": 0.4,
                "svm": 0.4,
                "tfidf": 0.2,
            }
            weights = {
                model: default_weights.get(model, 1.0)
                for model in self.base_models
            }
        if isinstance(weights, (list, tuple)):
            weights = {
                model: float(weights[idx])
                for idx, model in enumerate(self.base_models)
                if idx < len(weights)
            }
        if isinstance(weights, dict):
            normalized = {
                model: float(weights.get(model, 0.0))
                for model in self.base_models
            }
        else:
            normalized = {model: 1.0 for model in self.base_models}

        total = sum(max(value, 0.0) for value in normalized.values())
        if total <= 0:
            return {model: 1.0 / len(self.base_models) for model in self.base_models}
        return {
            model: max(value, 0.0) / total
            for model, value in normalized.items()
        }

    def analyze(self, text):
        model_results = {}
        for model_name, engine in self.engines.items():
            result = coerce_sentiment_result(engine.analyze(text), model_name)
            model_results[model_name] = result

        combined = {label: 0.0 for label in SENTIMENT_LABELS}
        for model_name, result in model_results.items():
            weight = self.weights.get(model_name, 0.0)
            for label in SENTIMENT_LABELS:
                combined[label] += weight * result.probs.get(label, 0.0)

        combined = normalize_probs(combined)
        sentiment = max(combined, key=combined.get)

        return SentimentResult(
            label=sentiment,
            score=float(combined.get(sentiment, 0.0)),
            probs=combined,
            model="ensemble",
            raw={
                "weights": self.weights,
                "models": {name: result.to_dict() for name, result in model_results.items()},
                "model_errors": self.model_errors,
            },
        )

    def batch_analyze(self, texts):
        model_outputs = {}
        for model_name, engine in self.engines.items():
            if hasattr(engine, "batch_analyze"):
                results = engine.batch_analyze(texts)
            else:
                results = [engine.analyze(text) for text in texts]
            model_outputs[model_name] = [
                coerce_sentiment_result(result, model_name)
                for result in results
            ]

        combined_results = []
        for idx in range(len(texts)):
            combined = {label: 0.0 for label in SENTIMENT_LABELS}
            for model_name, results in model_outputs.items():
                weight = self.weights.get(model_name, 0.0)
                result = results[idx]
                for label in SENTIMENT_LABELS:
                    combined[label] += weight * result.probs.get(label, 0.0)
            combined = normalize_probs(combined)
            sentiment = max(combined, key=combined.get)
            combined_results.append(SentimentResult(
                label=sentiment,
                score=float(combined.get(sentiment, 0.0)),
                probs=combined,
                model="ensemble",
                raw={
                    "weights": self.weights,
                    "models": {
                        name: model_outputs[name][idx].to_dict()
                        for name in model_outputs
                    },
                    "model_errors": self.model_errors,
                },
            ))

        return combined_results


class MetaLearnerSentimentEngine:

    def __init__(self, meta_model_path="./models/meta_learner.pkl", base_models=None):
        self.meta_model_path = _resolve_model_path(meta_model_path)
        if not self.meta_model_path.exists():
            raise FileNotFoundError(
                f"Meta-learner model not found: {self.meta_model_path}. "
                "Train and save using research/meta_learner.py"
            )

        with open(self.meta_model_path, "rb") as f:
            saved = pickle.load(f)

        self.meta_learner = saved.get("meta_learner")
        if self.meta_learner is None:
            raise RuntimeError(
                "Meta-learner file is missing the trained model. "
                "Re-train and save the meta-learner."
            )

        saved_base_models = saved.get("base_models") or []
        saved_base_models = [
            str(model).strip().lower()
            for model in saved_base_models
            if str(model).strip()
        ]
        if base_models is None:
            base_models = saved_base_models
            self.base_models_source = "model"
        else:
            self.base_models_source = "request"

        self.base_models = [
            str(model).strip().lower()
            for model in base_models
            if str(model).strip()
        ]
        if not self.base_models:
            raise RuntimeError(
                "Meta-learner base models are not specified. "
                "Provide base_models or retrain the meta-learner."
            )
        if saved_base_models and self.base_models != saved_base_models:
            raise RuntimeError(
                "Meta-learner base models do not match the trained configuration. "
                f"Saved order: {saved_base_models}. Requested: {self.base_models}."
            )

        self.feature_type = saved.get("feature_type", "probs")
        self.meta_learner_type = saved.get("meta_learner_type")
        self.label2idx = saved.get(
            "label2idx",
            {"Negative": 0, "Neutral": 1, "Positive": 2},
        )
        self.idx2label = saved.get(
            "idx2label",
            {value: key for key, value in self.label2idx.items()},
        )

        self.engines = {}
        self.model_errors = {}
        for model in self.base_models:
            if model in ("ensemble", "meta_learner", "stacking"):
                self.model_errors[model] = (
                    "Meta-learner base models cannot include ensemble/stacking."
                )
                continue
            try:
                self.engines[model] = get_base_engine(model)
            except Exception as exc:
                self.model_errors[model] = str(exc)

        missing = [model for model in self.base_models if model not in self.engines]
        if missing:
            raise RuntimeError(
                "Meta-learner base models could not be initialized: "
                f"{missing}. Errors: {self.model_errors}"
            )

    def _label_for_class(self, class_label):
        try:
            class_idx = int(class_label)
        except (TypeError, ValueError):
            return normalize_label(class_label)
        return self.idx2label.get(class_idx, normalize_label(class_label))

    def _feature_vector(self, result):
        probs = normalize_probs(result.probs)
        vector = [
            probs.get("Negative", 0.0),
            probs.get("Neutral", 0.0),
            probs.get("Positive", 0.0),
        ]
        if self.feature_type == "probs+logits":
            vector.append(float(result.score))
        return vector

    def _get_base_predictions(self, texts):
        model_outputs = {}
        for model_name, engine in self.engines.items():
            if hasattr(engine, "batch_analyze"):
                results = engine.batch_analyze(texts)
            else:
                results = [engine.analyze(text) for text in texts]
            model_outputs[model_name] = [
                coerce_sentiment_result(result, model_name)
                for result in results
            ]
        return model_outputs

    def _build_feature_matrix(self, base_predictions):
        if not base_predictions:
            return []
        n_samples = len(next(iter(base_predictions.values())))
        feature_matrix = []
        for idx in range(n_samples):
            row = []
            for model_name in self.base_models:
                row.extend(self._feature_vector(base_predictions[model_name][idx]))
            feature_matrix.append(row)
        return feature_matrix

    def _predict_proba(self, features):
        if hasattr(self.meta_learner, "predict_proba"):
            raw_probs = self.meta_learner.predict_proba(features)
            classes = getattr(
                self.meta_learner,
                "classes_",
                list(range(len(raw_probs[0]))),
            )
            probs_list = []
            for row in raw_probs:
                mapped = {}
                for idx, class_label in enumerate(classes):
                    mapped[self._label_for_class(class_label)] = float(row[idx])
                probs_list.append(normalize_probs(mapped))
            return probs_list

        predictions = self.meta_learner.predict(features)
        return [
            normalize_probs({self._label_for_class(pred): 1.0})
            for pred in predictions
        ]

    def analyze(self, text):
        results = self.batch_analyze([text])
        return results[0] if results else SentimentResult(
            label="Neutral",
            score=0.0,
            probs=normalize_probs({}),
            model="meta_learner",
            raw={"empty_input": True},
        )

    def batch_analyze(self, texts):
        if not texts:
            return []

        base_predictions = self._get_base_predictions(texts)
        features = self._build_feature_matrix(base_predictions)
        probs_list = self._predict_proba(features)

        results = []
        for probs in probs_list:
            label = max(probs, key=probs.get)
            results.append(SentimentResult(
                label=label,
                score=float(probs.get(label, 0.0)),
                probs=probs,
                model="meta_learner",
                raw={
                    "base_models": self.base_models,
                    "model_errors": self.model_errors,
                },
            ))
        return results


def get_sentiment_engine(engine_type='logreg', **kwargs):
    engines = {
        'tfidf': TFIDFSentimentEngine,
        'logreg': LogRegSentimentEngine,
        'svm': SVMSentimentEngine,
        'ensemble': EnsembleSentimentEngine,
        'ci_ensemble': EnsembleSentimentEngine,
        'meta_learner': MetaLearnerSentimentEngine,
        'stacking': MetaLearnerSentimentEngine,
        'hybrid_dl': None,  # Lazy loaded (requires PyTorch)
    }

    if engine_type not in engines:
        raise ValueError(
            f"Invalid engine type: {engine_type}. "
            f"Choose from: {list(engines.keys())}"
        )

    # Lazy import for hybrid_dl to avoid PyTorch dependency if not needed
    if engine_type == 'hybrid_dl':
        try:
            from .deep_models import HybridDLSentimentEngine
            return HybridDLSentimentEngine(**kwargs)
        except ImportError as e:
            raise ImportError(
                f"HybridDLSentimentEngine requires PyTorch. "
                f"Install with: pip install torch\n"
                f"Error: {e}"
            )

    return engines[engine_type](**kwargs)


def get_base_engine(engine_type='logreg', **kwargs):
    base_engines = {
        'tfidf': TFIDFSentimentEngine,
        'logreg': LogRegSentimentEngine,
        'svm': SVMSentimentEngine,
        'hybrid_dl': None,  # Lazy loaded (requires PyTorch)
    }

    if engine_type not in base_engines:
        raise ValueError(
            f"Invalid base engine type: {engine_type}. "
            f"Choose from: {list(base_engines.keys())}"
        )

    # Lazy import for hybrid_dl to avoid PyTorch dependency if not needed
    if engine_type == 'hybrid_dl':
        try:
            from .deep_models import HybridDLSentimentEngine
            return HybridDLSentimentEngine(**kwargs)
        except ImportError as e:
            raise ImportError(
                f"HybridDLSentimentEngine requires PyTorch. "
                f"Install with: pip install torch\n"
                f"Error: {e}"
            )

    return base_engines[engine_type](**kwargs)
BASE_DIR = Path(__file__).resolve().parents[1]


def _resolve_model_path(path_value):
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return (BASE_DIR / path_obj).resolve()
