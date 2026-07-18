"""
Meta-Learner (Stacking Ensemble) for Sentiment Analysis

Implements a 2-level stacked ensemble:
- Level 0: Base models (LogReg, SVM, TF-IDF, Hybrid-DL)
- Level 1: Meta-learner (Logistic Regression, XGBoost, etc.)

The meta-learner learns to optimally combine predictions from base models
using cross-validated out-of-fold predictions to avoid overfitting.

Author: Master's Thesis - Computational Intelligence
Reference: Wolpert (1992) - Stacked Generalization
"""

import sys
from pathlib import Path
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import pandas as pd

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.sentiment import get_sentiment_engine, coerce_sentiment_result, normalize_label
from src.utils import normalize_probs
from src.preprocessing import ClassicalPreprocessConfig, preprocess_classical_texts


class MetaLearnerEnsemble:
    """
    2-Level Stacked Ensemble for Sentiment Analysis

    Architecture:
        Level 0 (Base Models): LogReg, SVM, TF-IDF, Hybrid-DL
        Level 1 (Meta-Learner): Logistic Regression / XGBoost / LightGBM

    The meta-learner is trained on probability distributions from base models
    using cross-validated out-of-fold predictions to prevent overfitting.

    Features:
    - Multiple meta-learner options (LR, XGBoost, LightGBM)
    - Cross-validated training (k-fold)
    - Probability calibration
    - Uncertainty quantification

    Example:
        >>> meta = MetaLearnerEnsemble(
        ...     base_models=['logreg', 'svm', 'tfidf'],
        ...     meta_learner_type='logistic_regression'
        ... )
        >>> meta.fit(train_texts, train_labels)
        >>> predictions = meta.predict(test_texts)
    """

    def __init__(
        self,
        base_models=None,
        meta_learner_type='logistic_regression',
        n_folds=5,
        random_state=42,
        meta_params=None,
        feature_type='probs',  # 'probs' or 'probs+logits'
        base_training_mode: str = "per_fold",  # 'per_fold' (proper stacking) or 'pretrained' (legacy)
        preprocess: bool = False,
    ):
        """
        Args:
            base_models (list): List of base model names
            meta_learner_type (str): 'logistic_regression', 'xgboost', 'lightgbm'
            n_folds (int): Number of folds for cross-validated training
            random_state (int): Random seed
            meta_params (dict): Parameters for meta-learner
            feature_type (str): Type of features to use ('probs' or 'probs+logits')
        """
        if base_models is None:
            base_models = ['logreg', 'svm', 'tfidf']

        base_models = [
            str(model).strip().lower()
            for model in base_models
            if str(model).strip()
        ]
        self.base_models = base_models
        self.meta_learner_type = meta_learner_type
        self.n_folds = n_folds
        self.random_state = random_state
        self.feature_type = feature_type
        self.base_training_mode = str(base_training_mode or "per_fold").strip().lower()

        if self.base_training_mode not in {"per_fold", "pretrained"}:
            raise ValueError(
                "base_training_mode must be one of: per_fold, pretrained"
            )

        # Optional shared preprocessing used by classical models.
        # Keep this OFF by default to match existing baseline artifacts/training.
        self.enable_preprocessing = bool(preprocess)
        self.preprocess_config = ClassicalPreprocessConfig()

        # Vectorizer + base-model defaults (aligned with backend/train_*.py).
        self.vectorizer_params = {
            "lowercase": True,
            "max_features": 75000,
            "min_df": 2,
            "max_df": 0.95,
            "ngram_range": (1, 2),
            "strip_accents": None,
        }

        # In per-fold mode we fit these in-memory to avoid leakage.
        self._fitted_vectorizer = None
        self._fitted_base_models = None

        # Initialize base model engines (only required for legacy pretrained mode,
        # or for deployment-time usage with `src.sentiment.engines.meta_learner_engine`).
        self.engines = {}
        self.model_errors = {}

        if self.base_training_mode == "pretrained":
            for model_name in self.base_models:
                try:
                    self.engines[model_name] = get_sentiment_engine(model_name)
                    print(f"[OK] Loaded base model: {model_name}")
                except Exception as e:
                    self.model_errors[model_name] = str(e)
                    print(f"[ERROR] Failed to load {model_name}: {e}")

            if not self.engines:
                raise RuntimeError(
                    f"No base models could be loaded. Errors: {self.model_errors}"
                )

        # Initialize meta-learner
        self.meta_learner = self._create_meta_learner(meta_params)
        self.is_fitted = False

        # Label mapping
        self.label2idx = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.idx2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    def _create_meta_learner(self, params):
        """Create meta-learner classifier"""
        if params is None:
            params = {}

        if self.meta_learner_type == 'logistic_regression':
            return LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                class_weight=params.get('class_weight', 'balanced'),
                solver=params.get('solver', 'lbfgs'),
                random_state=self.random_state
            )

        elif self.meta_learner_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    max_depth=params.get('max_depth', 3),
                    n_estimators=params.get('n_estimators', 100),
                    learning_rate=params.get('learning_rate', 0.1),
                    subsample=params.get('subsample', 0.8),
                    colsample_bytree=params.get('colsample_bytree', 0.8),
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
            except ImportError:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        elif self.meta_learner_type == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    max_depth=params.get('max_depth', 3),
                    n_estimators=params.get('n_estimators', 100),
                    learning_rate=params.get('learning_rate', 0.1),
                    subsample=params.get('subsample', 0.8),
                    colsample_bytree=params.get('colsample_bytree', 0.8),
                    random_state=self.random_state
                )
            except ImportError:
                raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        else:
            raise ValueError(
                f"Unknown meta_learner_type: {self.meta_learner_type}. "
                "Choose from: logistic_regression, xgboost, lightgbm"
            )

    def _get_base_predictions(self, texts):
        """
        Get predictions from all base models

        Notes
        -----
        This method is used only when `base_training_mode='pretrained'`.
        In thesis-grade experiments, prefer `base_training_mode='per_fold'`
        which re-trains base models inside each fold to avoid leakage.

        Args:
            texts (list): List of text strings

        Returns:
            dict: {model_name: [SentimentResult objects]}
        """
        predictions = {}

        self._ensure_engines_loaded()

        for model_name, engine in self.engines.items():
            try:
                if hasattr(engine, 'batch_analyze'):
                    results = engine.batch_analyze(texts)
                else:
                    results = [engine.analyze(text) for text in texts]

                predictions[model_name] = [
                    coerce_sentiment_result(result, model_name)
                    for result in results
                ]
            except Exception as e:
                print(f"Warning: {model_name} prediction failed: {e}")
                # Create dummy predictions
                predictions[model_name] = [
                    coerce_sentiment_result({'label': 'Neutral'}, model_name)
                    for _ in texts
                ]

        return predictions

    def _ensure_engines_loaded(self) -> None:
        """Lazy-load base engines (used for prediction when in-memory models are absent)."""
        if self.engines:
            return
        for model_name in self.base_models:
            try:
                self.engines[model_name] = get_sentiment_engine(model_name)
                print(f"[OK] Loaded base model: {model_name}")
            except Exception as exc:
                self.model_errors[model_name] = str(exc)
                print(f"[ERROR] Failed to load {model_name}: {exc}")

        if not self.engines:
            raise RuntimeError(
                "No base models could be loaded for prediction. "
                f"Errors: {self.model_errors}"
            )

    def _fit_base_models(self, train_texts, train_labels):
        """
        Fit classical base models + a shared TF-IDF vectorizer on the given data.

        Returns
        -------
        (vectorizer, models_dict)
        """
        supported = {"logreg", "svm", "tfidf"}
        unknown = [m for m in self.base_models if m not in supported]
        if unknown:
            raise ValueError(
                "per_fold mode currently supports only classical base models "
                f"{sorted(supported)}. Unsupported: {unknown}"
            )

        train_texts = ["" if text is None else str(text) for text in train_texts]
        train_clean = (
            preprocess_classical_texts(train_texts, config=self.preprocess_config)
            if self.enable_preprocessing
            else train_texts
        )
        vectorizer = TfidfVectorizer(**self.vectorizer_params)
        X_train = vectorizer.fit_transform(train_clean)

        models = {}

        if "tfidf" in self.base_models:
            nb = MultinomialNB(alpha=0.5)
            nb.fit(X_train, train_labels)
            models["tfidf"] = nb

        if "logreg" in self.base_models:
            # Match backend/train_logreg_youtube.py defaults.
            import inspect

            lr_kwargs = {
                "C": 1.0,
                "max_iter": 200,
                "solver": "saga",
                "n_jobs": -1,
                "class_weight": None,
                "random_state": self.random_state,
            }
            if "multi_class" in inspect.signature(LogisticRegression).parameters:
                # Keep behavior consistent across scikit-learn versions.
                lr_kwargs["multi_class"] = "multinomial"

            lr = LogisticRegression(**lr_kwargs)
            lr.fit(X_train, train_labels)
            models["logreg"] = lr

        if "svm" in self.base_models:
            svm = LinearSVC(
                C=1.0,
                max_iter=2000,
                class_weight=None,
                random_state=self.random_state,
            )
            svm.fit(X_train, train_labels)
            models["svm"] = svm

        return vectorizer, models

    def _predict_proba_matrix(self, model_name, model, X_vec):
        """
        Return probabilities as a (n_samples, 3) matrix in (Negative, Neutral, Positive) order.
        """
        # Prefer calibrated probabilities when available.
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(X_vec)
            classes = getattr(model, "classes_", ["Negative", "Neutral", "Positive"])
            matrix = np.zeros((raw.shape[0], 3), dtype=float)
            for class_idx, label in enumerate(classes):
                norm = normalize_label(label)
                out_idx = self.label2idx.get(norm)
                if out_idx is not None:
                    matrix[:, out_idx] = raw[:, class_idx]
            return matrix

        # LinearSVC: use decision_function scores -> softmax (pseudo-probabilities).
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_vec)
            if getattr(scores, "ndim", 1) == 1:
                scores = np.vstack([-scores, scores]).T
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            classes = getattr(model, "classes_", ["Negative", "Neutral", "Positive"])
            matrix = np.zeros((probs.shape[0], 3), dtype=float)
            for class_idx, label in enumerate(classes):
                norm = normalize_label(label)
                out_idx = self.label2idx.get(norm)
                if out_idx is not None:
                    matrix[:, out_idx] = probs[:, class_idx]
            return matrix

        # Fallback: uniform
        return np.full((X_vec.shape[0], 3), 1.0 / 3.0, dtype=float)

    def _base_feature_matrix(self, vectorizer, models, texts):
        """
        Build Level-0 feature matrix from fitted base models for the given texts.
        """
        texts = ["" if text is None else str(text) for text in texts]
        texts_clean = (
            preprocess_classical_texts(texts, config=self.preprocess_config)
            if self.enable_preprocessing
            else texts
        )
        X = vectorizer.transform(texts_clean)

        n_samples = X.shape[0]
        per_model = 3 if self.feature_type == "probs" else 4
        features = np.zeros((n_samples, len(self.base_models) * per_model), dtype=float)

        for model_idx, model_name in enumerate(self.base_models):
            model = models.get(model_name)
            if model is None:
                raise RuntimeError(f"Missing fitted base model: {model_name}")
            probs = self._predict_proba_matrix(model_name, model, X)
            start = model_idx * per_model
            features[:, start : start + 3] = probs
            if self.feature_type == "probs+logits":
                features[:, start + 3] = probs.max(axis=1)

        return features

    def _extract_features(self, base_predictions):
        """
        Extract features from base model predictions

        Args:
            base_predictions (dict): {model_name: [results]}

        Returns:
            np.ndarray: Feature matrix (n_samples, n_features)
        """
        n_samples = len(list(base_predictions.values())[0])
        model_order = [
            model_name
            for model_name in self.base_models
            if model_name in base_predictions
        ]

        features = []
        for idx in range(n_samples):
            row = []
            for model_name in model_order:
                result = base_predictions[model_name][idx]
                probs = normalize_probs(result.probs)

                if self.feature_type == 'probs':
                    feature_vector = [
                        probs.get('Negative', 0.0),
                        probs.get('Neutral', 0.0),
                        probs.get('Positive', 0.0)
                    ]
                elif self.feature_type == 'probs+logits':
                    feature_vector = [
                        probs.get('Negative', 0.0),
                        probs.get('Neutral', 0.0),
                        probs.get('Positive', 0.0),
                        result.score
                    ]
                else:
                    feature_vector = [
                        probs.get('Negative', 0.0),
                        probs.get('Neutral', 0.0),
                        probs.get('Positive', 0.0)
                    ]

                row.extend(feature_vector)
            features.append(row)

        return np.array(features)

    def fit(self, texts, labels, verbose=True):
        """
        Train meta-learner using cross-validated Level-0 predictions.

        Thesis-grade default (`base_training_mode='per_fold'`)
        ------------------------------------------------------
        For each fold, base models + TF-IDF vectorizer are trained on the
        fold's training split, and predictions are generated on the held-out
        split. The meta-learner is then trained on these out-of-fold features.

        Legacy mode (`base_training_mode='pretrained'`)
        ------------------------------------------------
        Uses already-trained base engines from disk. This can leak information
        if those engines were trained using the same data supplied here.

        Args:
            texts (list): Training texts
            labels (list): Training labels (Positive/Neutral/Negative)
            verbose (bool): Print progress

        Returns:
            self
        """
        if verbose:
            print("\n" + "="*80)
            print("TRAINING META-LEARNER ENSEMBLE")
            print("="*80)
            print(f"Base models: {self.base_models}")
            print(f"Meta-learner: {self.meta_learner_type}")
            print(f"Base training mode: {self.base_training_mode}")
            print(f"Training samples: {len(texts)}")
            print(f"Cross-validation folds: {self.n_folds}")

        # Normalize labels
        labels = [normalize_label(label) for label in labels]
        label_indices = np.array([self.label2idx[label] for label in labels])

        # Step 1: Generate out-of-fold predictions from base models
        if verbose:
            print("\n[INFO] Generating out-of-fold predictions from base models...")

        per_model = 3 if self.feature_type == "probs" else 4
        oof_features = np.zeros((len(texts), len(self.base_models) * per_model))  # probs (+ optional max prob)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, label_indices)):
            if verbose:
                print(f"   Fold {fold+1}/{self.n_folds}: {len(val_idx)} validation samples")

            val_texts = [texts[i] for i in val_idx]

            if self.base_training_mode == "pretrained":
                if verbose and fold == 0:
                    print(
                        "   ⚠️  Using pretrained base engines. If they were trained on this same dataset, "
                        "the resulting meta-learner will be optimistically biased."
                    )
                base_preds = self._get_base_predictions(val_texts)
                fold_features = self._extract_features(base_preds)
            else:
                fold_train_texts = [texts[i] for i in train_idx]
                fold_train_labels = [labels[i] for i in train_idx]
                vectorizer, models = self._fit_base_models(fold_train_texts, fold_train_labels)
                fold_features = self._base_feature_matrix(vectorizer, models, val_texts)

            # Store in out-of-fold matrix
            oof_features[val_idx] = fold_features

        # Step 2: Train meta-learner on out-of-fold predictions
        if verbose:
            print("\n[INFO] Training meta-learner...")

        self.meta_learner.fit(oof_features, label_indices)

        self.is_fitted = True

        # In thesis-grade mode, fit base models on full training data for
        # downstream prediction/evaluation without requiring pre-trained artifacts.
        if self.base_training_mode == "per_fold":
            self._fitted_vectorizer, self._fitted_base_models = self._fit_base_models(
                texts, labels
            )

        if verbose:
            # Evaluate on training data (out-of-fold predictions)
            train_preds = self.meta_learner.predict(oof_features)
            train_acc = accuracy_score(label_indices, train_preds)
            train_f1 = f1_score(label_indices, train_preds, average='macro')

            print("\n[OK] Meta-learner trained successfully")
            print(f"   Training accuracy (OOF): {train_acc:.4f}")
            print(f"   Training F1 (OOF):      {train_f1:.4f}")

        return self

    def predict(self, texts):
        """
        Predict sentiment for texts using meta-learner

        Args:
            texts (list): List of text strings

        Returns:
            list: Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-learner not fitted. Call fit() first.")

        # Prefer in-memory fitted base models (available after `fit()` in per_fold mode).
        if self._fitted_vectorizer is not None and self._fitted_base_models is not None:
            features = self._base_feature_matrix(
                self._fitted_vectorizer,
                self._fitted_base_models,
                texts,
            )
        else:
            # Get base model predictions from pre-trained engines
            base_preds = self._get_base_predictions(texts)
            features = self._extract_features(base_preds)

        # Meta-learner prediction
        pred_indices = self.meta_learner.predict(features)
        predictions = [self.idx2label[idx] for idx in pred_indices]

        return predictions

    def predict_proba(self, texts):
        """
        Predict probability distributions using meta-learner

        Args:
            texts (list): List of text strings

        Returns:
            list: List of probability dictionaries
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-learner not fitted. Call fit() first.")

        if self._fitted_vectorizer is not None and self._fitted_base_models is not None:
            features = self._base_feature_matrix(
                self._fitted_vectorizer,
                self._fitted_base_models,
                texts,
            )
        else:
            base_preds = self._get_base_predictions(texts)
            features = self._extract_features(base_preds)

        # Meta-learner probability prediction
        probs = self.meta_learner.predict_proba(features)

        # Convert to list of dicts
        probs_list = []
        for prob_vector in probs:
            probs_dict = {
                'Negative': float(prob_vector[0]),
                'Neutral': float(prob_vector[1]),
                'Positive': float(prob_vector[2])
            }
            probs_list.append(normalize_probs(probs_dict))

        return probs_list

    def evaluate(self, texts, labels, verbose=True):
        """
        Evaluate meta-learner on test data

        Args:
            texts (list): Test texts
            labels (list): True labels
            verbose (bool): Print results

        Returns:
            dict: Evaluation metrics
        """
        labels = [normalize_label(label) for label in labels]
        predictions = self.predict(texts)

        label_indices = [self.label2idx[label] for label in labels]
        pred_indices = [self.label2idx[pred] for pred in predictions]

        metrics = {
            'accuracy': accuracy_score(label_indices, pred_indices),
            'f1_macro': f1_score(label_indices, pred_indices, average='macro'),
            'classification_report': classification_report(
                label_indices, pred_indices,
                target_names=['Negative', 'Neutral', 'Positive'],
                output_dict=True
            )
        }

        if verbose:
            print("\n" + "="*80)
            print("META-LEARNER EVALUATION")
            print("="*80)
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"F1 (macro): {metrics['f1_macro']:.4f}")
            print("\nPer-class metrics:")
            print(classification_report(
                label_indices, pred_indices,
                target_names=['Negative', 'Neutral', 'Positive']
            ))

        return metrics

    def save(self, path):
        """Save meta-learner to disk"""
        save_data = {
            'base_models': self.base_models,
            'base_training_mode': self.base_training_mode,
            'meta_learner_type': self.meta_learner_type,
            'meta_learner': self.meta_learner,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'feature_type': self.feature_type,
            'vectorizer_params': self.vectorizer_params,
            'preprocessing': {
                "enabled": bool(self.enable_preprocessing),
                "type": "classical_v1",
                "config": {
                    "expand_negation_contractions": self.preprocess_config.expand_negation_contractions,
                    "negation_tag": self.preprocess_config.negation_tag,
                    "remove_stopwords": self.preprocess_config.remove_stopwords,
                },
            },
            'label2idx': self.label2idx,
            'idx2label': self.idx2label,
            'is_fitted': self.is_fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"[OK] Meta-learner saved to: {path}")

    @classmethod
    def load(cls, path):
        """Load meta-learner from disk"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        instance = cls(
            base_models=save_data['base_models'],
            meta_learner_type=save_data['meta_learner_type'],
            n_folds=save_data['n_folds'],
            random_state=save_data['random_state'],
            feature_type=save_data.get('feature_type', 'probs'),
            base_training_mode=save_data.get('base_training_mode', 'pretrained'),
            preprocess=bool((save_data.get("preprocessing") or {}).get("enabled", False)),
        )

        instance.meta_learner = save_data['meta_learner']
        instance.vectorizer_params = save_data.get('vectorizer_params', instance.vectorizer_params)
        instance.label2idx = save_data['label2idx']
        instance.idx2label = save_data['idx2label']
        instance.is_fitted = save_data['is_fitted']

        print(f"[OK] Meta-learner loaded from: {path}")
        return instance


# Command-line interface
def main():
    """CLI for training and evaluating meta-learner"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train and evaluate meta-learner ensemble'
    )
    parser.add_argument(
        '--data',
        required=False,
        help=(
            "Path to a labeled CSV dataset with 'text' and 'label' columns. "
            "If provided, the script will create an internal train/test split. "
            "Prefer --train_csv/--test_csv for thesis-grade reproducibility."
        ),
    )
    parser.add_argument(
        '--train_csv',
        required=False,
        help="Explicit training split CSV (expects 'text' and 'label' columns by default).",
    )
    parser.add_argument(
        '--test_csv',
        required=False,
        help="Explicit test split CSV (expects 'text' and 'label' columns by default).",
    )
    parser.add_argument('--text_column', default='text', help='Name of the text column')
    parser.add_argument('--label_column', default='label', help='Name of the label column')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument(
        '--base_models',
        default='logreg,svm,tfidf',
        help='Comma-separated base models'
    )
    parser.add_argument(
        '--meta_learner',
        default='logistic_regression',
        choices=['logistic_regression', 'xgboost', 'lightgbm'],
        help='Meta-learner type'
    )
    parser.add_argument(
        '--base_training_mode',
        default='per_fold',
        choices=['per_fold', 'pretrained'],
        help=(
            "How to generate Level-0 features for stacking. "
            "'per_fold' retrains base models inside each fold (recommended, avoids leakage). "
            "'pretrained' uses existing saved engines (legacy; can leak if trained on same data)."
        ),
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help=(
            "Enable the shared classical preprocessing for training/evaluating the base models "
            "inside the stacking pipeline. Keep this OFF unless your baseline models are trained "
            "with preprocessing enabled."
        ),
    )
    parser.add_argument('--n_folds', type=int, default=5, help='CV folds for training')
    parser.add_argument('--output', help='Path to save trained meta-learner')

    args = parser.parse_args()

    def _load_split(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if args.text_column not in df.columns or args.label_column not in df.columns:
            raise ValueError(
                f"Dataset must contain '{args.text_column}' and '{args.label_column}' columns. "
                f"Got columns: {list(df.columns)}"
            )
        df = df.dropna(subset=[args.text_column, args.label_column])
        df = df.rename(columns={args.text_column: "text", args.label_column: "label"})
        return df

    # Load data
    if args.train_csv:
        if not args.test_csv:
            raise ValueError("--test_csv is required when using --train_csv")
        print("Loading predefined splits...")
        train_df = _load_split(args.train_csv)
        test_df = _load_split(args.test_csv)
    else:
        if not args.data:
            raise ValueError("Provide either --data or --train_csv/--test_csv")

        print("Loading dataset...")
        df = _load_split(args.data)

        # Split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=df['label'],
        )

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    print(f"Train: {len(train_texts)} samples")
    print(f"Test:  {len(test_texts)} samples")

    # Create and train meta-learner
    base_models = [m.strip() for m in args.base_models.split(',')]

    meta = MetaLearnerEnsemble(
        base_models=base_models,
        meta_learner_type=args.meta_learner,
        n_folds=args.n_folds,
        random_state=args.random_seed,
        base_training_mode=args.base_training_mode,
        preprocess=args.preprocess,
    )

    meta.fit(train_texts, train_labels)

    # Evaluate
    metrics = meta.evaluate(test_texts, test_labels)

    # Save if requested
    if args.output:
        meta.save(args.output)


if __name__ == "__main__":
    main()
