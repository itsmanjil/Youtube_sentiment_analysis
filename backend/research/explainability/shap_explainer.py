"""
SHAP-based Model Explainer.

This module provides SHAP (SHapley Additive exPlanations) for
explaining sentiment analysis predictions.

Mathematical Foundation
-----------------------
SHAP values are based on Shapley values from cooperative game theory.
For a model f and input x with features {1, 2, ..., p}:

    phi_i(f, x) = sum over S subset of {1,...,p}\{i}:
        |S|!(p-|S|-1)!/p! * [f(S union {i}) - f(S)]

Where:
    - phi_i is the SHAP value for feature i
    - f(S) is the model prediction using only features in S
    - The sum considers all possible feature subsets

Properties of SHAP:
1. Local accuracy: sum(phi_i) + phi_0 = f(x)
2. Missingness: phi_i = 0 for features not in input
3. Consistency: If feature contribution increases, SHAP value increases

For text classification, KernelSHAP is used, which approximates
Shapley values using a weighted linear regression.

References
----------
Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting
model predictions. NeurIPS.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path


class SHAPExplainer:
    """
    SHAP-based explainer for sentiment analysis models.

    This class wraps the SHAP library to provide model-agnostic
    explanations for text classification.

    Parameters
    ----------
    model : Any
        Trained sentiment analysis model with predict method.
    vectorizer : Any, optional
        TF-IDF or other text vectorizer.
        Required for classical ML models.
    predict_fn : Callable, optional
        Custom prediction function. If not provided, uses model.predict_proba
        or model.analyze for custom engines.
    background_data : array-like, optional
        Background dataset for SHAP. If not provided, creates a synthetic
        background using the masker.
    class_names : List[str], optional
        Names of the output classes.
        Default: ['Negative', 'Neutral', 'Positive']

    Attributes
    ----------
    explainer : shap.Explainer
        SHAP explainer object.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>>
    >>> # Train model
    >>> vectorizer = TfidfVectorizer()
    >>> X = vectorizer.fit_transform(texts)
    >>> model = LogisticRegression().fit(X, labels)
    >>>
    >>> # Create explainer
    >>> explainer = SHAPExplainer(model, vectorizer)
    >>>
    >>> # Explain a prediction
    >>> explanation = explainer.explain("This video is amazing!")
    >>> print(explanation['shap_values'])

    Notes
    -----
    SHAP requires the 'shap' package. Install with:
        pip install shap

    For transformer models, use the Partition explainer or
    IntegratedGradientsExplainer for better performance.
    """

    def __init__(
        self,
        model: Any,
        vectorizer: Optional[Any] = None,
        predict_fn: Optional[Callable] = None,
        background_data: Optional[Any] = None,
        class_names: Optional[List[str]] = None,
    ):
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAPExplainer requires the shap library. "
                "Install with: pip install shap"
            )

        self.shap = shap
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = class_names or ["Negative", "Neutral", "Positive"]

        # Create prediction function
        if predict_fn is not None:
            self.predict_fn = predict_fn
        elif vectorizer is not None:
            # Classical ML model with vectorizer
            self.predict_fn = self._create_sklearn_predict_fn()
        elif hasattr(model, "batch_analyze"):
            # Custom sentiment engine
            self.predict_fn = self._create_engine_predict_fn()
        elif hasattr(model, "predict_proba"):
            # Generic sklearn model
            self.predict_fn = model.predict_proba
        else:
            raise ValueError(
                "Could not infer prediction function. "
                "Please provide a predict_fn argument."
            )

        # Create SHAP explainer
        if vectorizer is not None:
            # For text models, use a text masker
            self.masker = shap.maskers.Text(tokenizer=r"\W+")
            self.explainer = shap.Explainer(
                self.predict_fn,
                self.masker,
                output_names=self.class_names,
            )
        else:
            # For pre-vectorized data
            self.explainer = shap.KernelExplainer(
                self.predict_fn,
                background_data if background_data is not None else np.zeros((1, 100)),
            )

    def _create_sklearn_predict_fn(self) -> Callable:
        """Create prediction function for sklearn models with vectorizer."""
        def predict(texts):
            if isinstance(texts, str):
                texts = [texts]
            X = self.vectorizer.transform(texts)
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            else:
                # For models without predict_proba
                predictions = self.model.predict(X)
                # Convert to one-hot probabilities
                n_classes = len(self.class_names)
                probs = np.zeros((len(predictions), n_classes))
                for i, pred in enumerate(predictions):
                    probs[i, int(pred)] = 1.0
                return probs
        return predict

    def _create_engine_predict_fn(self) -> Callable:
        """Create prediction function for custom sentiment engines."""
        def predict(texts):
            if isinstance(texts, str):
                texts = [texts]
            results = self.model.batch_analyze(texts)
            probs = np.array([
                [r.probs.get("Negative", 0), r.probs.get("Neutral", 0), r.probs.get("Positive", 0)]
                for r in results
            ])
            return probs
        return predict

    def explain(
        self,
        text: str,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single text.

        Parameters
        ----------
        text : str
            Text to explain.
        target_class : int, optional
            Class index to explain. If None, explains the predicted class.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - text: Original text
            - tokens: Tokenized text
            - shap_values: SHAP values per token per class
            - base_value: Expected model output (baseline)
            - prediction: Model prediction
            - predicted_class: Predicted class index
            - predicted_probs: Class probabilities
        """
        # Get SHAP values
        shap_values = self.explainer([text])

        # Get prediction
        probs = self.predict_fn([text])[0]
        predicted_class = int(np.argmax(probs))

        if target_class is None:
            target_class = predicted_class

        # Extract values
        if hasattr(shap_values, "values"):
            values = shap_values.values[0]
            base_value = shap_values.base_values[0]
            data = shap_values.data[0] if hasattr(shap_values, "data") else text.split()
        else:
            values = shap_values[0]
            base_value = 0.0
            data = text.split()

        # Convert to list for JSON serialization
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if isinstance(base_value, np.ndarray):
            base_value = base_value.tolist()
        if isinstance(data, np.ndarray):
            data = data.tolist()

        return {
            "text": text,
            "tokens": data,
            "shap_values": values,
            "base_value": base_value,
            "prediction": self.class_names[predicted_class],
            "predicted_class": predicted_class,
            "predicted_probs": probs.tolist(),
            "target_class": target_class,
            "target_class_name": self.class_names[target_class],
        }

    def explain_batch(
        self,
        texts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Generate SHAP explanations for multiple texts.

        Parameters
        ----------
        texts : List[str]
            Texts to explain.

        Returns
        -------
        List[Dict[str, Any]]
            List of explanation dictionaries.
        """
        return [self.explain(text) for text in texts]

    def get_feature_importance(
        self,
        texts: List[str],
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """
        Compute global feature importance from multiple texts.

        Parameters
        ----------
        texts : List[str]
            Corpus of texts to analyze.
        top_n : int, optional
            Number of top features to return per class.
            Default: 20

        Returns
        -------
        Dict[str, Any]
            Dictionary with feature importance per class.
        """
        # Get SHAP values for all texts
        all_shap_values = []
        all_tokens = []

        for text in texts:
            explanation = self.explain(text)
            all_shap_values.append(explanation["shap_values"])
            all_tokens.extend(explanation["tokens"])

        # Aggregate by token
        token_importance = {class_name: {} for class_name in self.class_names}

        for explanation in all_shap_values:
            values = explanation["shap_values"]
            tokens = explanation.get("tokens", [])

            for i, token in enumerate(tokens):
                if i < len(values):
                    for class_idx, class_name in enumerate(self.class_names):
                        if class_idx < len(values[i]) if isinstance(values[i], list) else True:
                            val = values[i][class_idx] if isinstance(values[i], list) else values[i]
                            if token not in token_importance[class_name]:
                                token_importance[class_name][token] = []
                            token_importance[class_name][token].append(val)

        # Average and rank
        result = {}
        for class_name in self.class_names:
            avg_importance = {
                token: np.mean(vals)
                for token, vals in token_importance[class_name].items()
                if vals
            }
            sorted_tokens = sorted(
                avg_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]
            result[class_name] = {
                "positive": [(t, v) for t, v in sorted_tokens if v > 0][:top_n // 2],
                "negative": [(t, v) for t, v in sorted_tokens if v < 0][:top_n // 2],
            }

        return result

    def visualize(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Visualize SHAP explanation.

        Parameters
        ----------
        explanation : Dict[str, Any]
            Explanation from explain() method.
        save_path : str or Path, optional
            Path to save the visualization. If None, displays inline.
        """
        import matplotlib.pyplot as plt

        tokens = explanation["tokens"]
        values = explanation["shap_values"]
        target_class = explanation["target_class"]

        # Extract values for target class
        if isinstance(values[0], list):
            class_values = [v[target_class] for v in values]
        else:
            class_values = values

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, max(6, len(tokens) * 0.3)))

        colors = ["green" if v > 0 else "red" for v in class_values]
        y_pos = np.arange(len(tokens))

        ax.barh(y_pos, class_values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens)
        ax.set_xlabel("SHAP Value")
        ax.set_title(
            f"SHAP Explanation for '{explanation['prediction']}' prediction\n"
            f"(Target class: {explanation['target_class_name']})"
        )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
