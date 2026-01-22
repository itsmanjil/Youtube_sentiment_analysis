"""
LIME-based Model Explainer.

This module provides LIME (Local Interpretable Model-agnostic Explanations)
for explaining sentiment analysis predictions.

Mathematical Foundation
-----------------------
LIME explains a prediction by learning a simple interpretable model
(linear regression) locally around the prediction.

For input x and complex model f:

1. Generate perturbed samples around x by randomly removing words
2. Get model predictions for perturbed samples
3. Weight samples by similarity to x: pi_x(z) = exp(-D(x,z)^2 / sigma^2)
4. Fit weighted linear regression: g(z') = argmin sum pi_x(z) (f(z) - g(z'))^2

The coefficients of g indicate feature importance.

Advantages over SHAP:
- Faster computation (local approximation)
- Works well with any black-box model
- Easy to interpret linear explanations

Disadvantages:
- Explanations may not be globally consistent
- Sensitive to perturbation strategy

References
----------
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?":
Explaining the predictions of any classifier. KDD.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path


class LIMEExplainer:
    """
    LIME-based explainer for sentiment analysis models.

    LIME generates local explanations by learning a simple interpretable
    model around each prediction.

    Parameters
    ----------
    predict_fn : Callable
        Function that takes list of texts and returns class probabilities.
        Signature: (texts: List[str]) -> np.ndarray of shape (n, n_classes)
    class_names : List[str], optional
        Names of the output classes.
        Default: ['Negative', 'Neutral', 'Positive']
    kernel_width : float, optional
        Width of the exponential kernel for weighting samples.
        Default: 25

    Attributes
    ----------
    explainer : LimeTextExplainer
        LIME text explainer object.

    Examples
    --------
    >>> def predict_fn(texts):
    ...     # Your model prediction logic
    ...     return model.predict_proba(texts)
    >>>
    >>> explainer = LIMEExplainer(predict_fn)
    >>> explanation = explainer.explain("This video is amazing!")
    >>> print(explanation['feature_weights'])

    Notes
    -----
    LIME requires the 'lime' package. Install with:
        pip install lime
    """

    def __init__(
        self,
        predict_fn: Callable,
        class_names: Optional[List[str]] = None,
        kernel_width: float = 25,
    ):
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            raise ImportError(
                "LIMEExplainer requires the lime library. "
                "Install with: pip install lime"
            )

        self.predict_fn = predict_fn
        self.class_names = class_names or ["Negative", "Neutral", "Positive"]

        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            kernel_width=kernel_width,
            bow=True,  # Bag-of-words representation
        )

    @classmethod
    def from_sentiment_engine(
        cls,
        engine: Any,
        class_names: Optional[List[str]] = None,
    ) -> "LIMEExplainer":
        """
        Create LIME explainer from a sentiment engine.

        Parameters
        ----------
        engine : BaseSentimentEngine
            Sentiment analysis engine with batch_analyze method.
        class_names : List[str], optional
            Class names.

        Returns
        -------
        LIMEExplainer
            Configured LIME explainer.
        """
        def predict_fn(texts):
            if isinstance(texts, str):
                texts = [texts]
            results = engine.batch_analyze(texts)
            probs = np.array([
                [r.probs.get("Negative", 0), r.probs.get("Neutral", 0), r.probs.get("Positive", 0)]
                for r in results
            ])
            return probs

        return cls(predict_fn, class_names)

    @classmethod
    def from_sklearn_model(
        cls,
        model: Any,
        vectorizer: Any,
        class_names: Optional[List[str]] = None,
    ) -> "LIMEExplainer":
        """
        Create LIME explainer from sklearn model with vectorizer.

        Parameters
        ----------
        model : sklearn classifier
            Trained classifier with predict_proba method.
        vectorizer : TfidfVectorizer or similar
            Fitted text vectorizer.
        class_names : List[str], optional
            Class names.

        Returns
        -------
        LIMEExplainer
            Configured LIME explainer.
        """
        def predict_fn(texts):
            if isinstance(texts, str):
                texts = [texts]
            X = vectorizer.transform(texts)
            return model.predict_proba(X)

        return cls(predict_fn, class_names)

    def explain(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 5000,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single text.

        Parameters
        ----------
        text : str
            Text to explain.
        num_features : int, optional
            Number of features to include in explanation.
            Default: 10
        num_samples : int, optional
            Number of perturbed samples to generate.
            Default: 5000
        target_class : int, optional
            Class index to explain. If None, explains predicted class.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - text: Original text
            - prediction: Predicted class name
            - predicted_class: Predicted class index
            - predicted_probs: Class probabilities
            - feature_weights: List of (word, weight) tuples
            - intercept: Local model intercept
            - local_prediction: Local model prediction
        """
        # Get model prediction
        probs = self.predict_fn([text])[0]
        predicted_class = int(np.argmax(probs))

        if target_class is None:
            target_class = predicted_class

        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            text,
            self.predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=[target_class],
        )

        # Extract feature weights
        feature_weights = exp.as_list(label=target_class)

        # Get local model details
        local_exp = exp.local_exp[target_class]

        return {
            "text": text,
            "prediction": self.class_names[predicted_class],
            "predicted_class": predicted_class,
            "predicted_probs": probs.tolist(),
            "target_class": target_class,
            "target_class_name": self.class_names[target_class],
            "feature_weights": feature_weights,
            "intercept": float(exp.intercept[target_class]),
            "local_prediction": float(exp.local_pred[target_class]),
            "score": float(exp.score),
        }

    def explain_batch(
        self,
        texts: List[str],
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Generate LIME explanations for multiple texts.

        Parameters
        ----------
        texts : List[str]
            Texts to explain.
        num_features : int, optional
            Number of features per explanation.
        num_samples : int, optional
            Number of perturbed samples.

        Returns
        -------
        List[Dict[str, Any]]
            List of explanation dictionaries.
        """
        return [
            self.explain(text, num_features, num_samples)
            for text in texts
        ]

    def get_top_features(
        self,
        explanation: Dict[str, Any],
        n: int = 5,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Get top positive and negative features from explanation.

        Parameters
        ----------
        explanation : Dict[str, Any]
            Explanation from explain() method.
        n : int, optional
            Number of top features to return per direction.
            Default: 5

        Returns
        -------
        Tuple[List, List]
            (positive_features, negative_features)
        """
        weights = explanation["feature_weights"]

        positive = [(w, v) for w, v in weights if v > 0]
        negative = [(w, v) for w, v in weights if v < 0]

        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: x[1])

        return positive[:n], negative[:n]

    def visualize(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Visualize LIME explanation as a bar plot.

        Parameters
        ----------
        explanation : Dict[str, Any]
            Explanation from explain() method.
        save_path : str or Path, optional
            Path to save the visualization.
        """
        import matplotlib.pyplot as plt

        weights = explanation["feature_weights"]
        words = [w for w, _ in weights]
        values = [v for _, v in weights]

        # Sort by absolute value
        sorted_indices = np.argsort(np.abs(values))[::-1]
        words = [words[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(words) * 0.4)))

        colors = ["green" if v > 0 else "red" for v in values]
        y_pos = np.arange(len(words))

        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.set_xlabel("Feature Weight")
        ax.set_title(
            f"LIME Explanation for '{explanation['prediction']}' prediction\n"
            f"(Score: {explanation['score']:.3f})"
        )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_html(
        self,
        text: str,
        num_features: int = 10,
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate HTML visualization with highlighted text.

        Parameters
        ----------
        text : str
            Text to explain.
        num_features : int, optional
            Number of features.
        save_path : str or Path, optional
            Path to save HTML file.

        Returns
        -------
        str
            HTML string with visualization.
        """
        probs = self.predict_fn([text])[0]
        predicted_class = int(np.argmax(probs))

        exp = self.explainer.explain_instance(
            text,
            self.predict_fn,
            num_features=num_features,
            labels=[predicted_class],
        )

        html = exp.as_html()

        if save_path:
            with open(save_path, "w") as f:
                f.write(html)

        return html
