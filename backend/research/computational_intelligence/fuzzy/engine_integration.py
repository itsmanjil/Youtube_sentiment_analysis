"""
Integration Layer for Fuzzy Sentiment with Existing Engines

This module provides seamless integration between the fuzzy sentiment
classification system and the existing sentiment engines in the project.

Key Features:
    - Automatic adapter for existing engines (TF-IDF, LogReg, SVM, etc.)
    - Batch processing support
    - Result format compatibility with existing API
    - Performance monitoring and logging

Usage:
    >>> from app.sentiment_engines import LogRegSentimentEngine, SVMSentimentEngine
    >>> from research.computational_intelligence.fuzzy import FuzzySentimentEngine
    >>>
    >>> # Initialize fuzzy engine with existing engines
    >>> fuzzy_engine = FuzzySentimentEngine.from_existing_engines(
    ...     engine_configs=[
    ...         {'name': 'logreg', 'class': 'LogRegSentimentEngine'},
    ...         {'name': 'svm', 'class': 'SVMSentimentEngine'},
    ...     ]
    ... )
    >>>
    >>> # Use like any other sentiment engine
    >>> result = fuzzy_engine.analyze("This video is amazing!")

Author: [Your Name]
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass
import time
import logging

# Add parent paths for imports
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from .fuzzy_sentiment import FuzzySentimentClassifier, FuzzySentimentResult
from .fuzzy_evaluation import FuzzyEvaluator


logger = logging.getLogger(__name__)


@dataclass
class FuzzyAnalysisResult:
    """
    Result format compatible with existing SentimentResult.

    Extends the standard result with fuzzy-specific fields while
    maintaining backward compatibility.
    """
    label: str
    score: float
    probs: Dict[str, float]
    model: str
    raw: Dict[str, Any] = None

    # Fuzzy-specific fields
    uncertainty: float = 0.0
    confidence: float = 1.0
    fuzziness_index: float = 0.0
    base_model_scores: Dict[str, float] = None
    defuzz_method: str = 'centroid'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'label': self.label,
            'score': self.score,
            'probs': self.probs,
            'model': self.model,
            'raw': self.raw,
            'uncertainty': self.uncertainty,
            'confidence': self.confidence,
            'fuzziness_index': self.fuzziness_index,
            'base_model_scores': self.base_model_scores,
            'defuzz_method': self.defuzz_method,
        }

    @classmethod
    def from_fuzzy_result(
        cls,
        result: FuzzySentimentResult,
        model_name: str = 'fuzzy_ensemble'
    ) -> 'FuzzyAnalysisResult':
        """Create from FuzzySentimentResult."""
        return cls(
            label=result.label,
            score=result.crisp_score,
            probs=result.probabilities,
            model=model_name,
            raw=result.to_dict(),
            uncertainty=result.uncertainty_metrics.get('fuzziness', 0),
            confidence=result.confidence,
            fuzziness_index=result.uncertainty_metrics.get('fuzziness', 0),
            base_model_scores=result.metadata.get('input_scores', {}),
            defuzz_method=result.metadata.get('defuzz_method', 'centroid'),
        )


class EngineAdapter:
    """
    Adapter to standardize interface for different sentiment engine types.

    Handles various engine interfaces:
    - analyze(text) -> result
    - predict(text) -> label
    - predict_proba(text) -> probabilities
    """

    def __init__(self, engine: Any, name: str):
        """
        Initialize adapter for a sentiment engine.

        Parameters
        ----------
        engine : Any
            Sentiment engine instance
        name : str
            Name identifier for this engine
        """
        self.engine = engine
        self.name = name
        self._detect_interface()

    def _detect_interface(self) -> None:
        """Detect which interface the engine supports."""
        self.has_analyze = hasattr(self.engine, 'analyze')
        self.has_predict_proba = hasattr(self.engine, 'predict_proba')
        self.has_predict = hasattr(self.engine, 'predict')

        if not any([self.has_analyze, self.has_predict_proba, self.has_predict]):
            raise ValueError(
                f"Engine '{self.name}' must have 'analyze', 'predict_proba', "
                f"or 'predict' method"
            )

    def get_probabilities(self, text: str) -> Dict[str, float]:
        """
        Get class probabilities from the engine.

        Parameters
        ----------
        text : str
            Input text to classify

        Returns
        -------
        dict
            Dictionary with 'Positive', 'Negative', 'Neutral' probabilities
        """
        if self.has_analyze:
            result = self.engine.analyze(text)
            if hasattr(result, 'probs'):
                return self._normalize_probs(result.probs)
            elif isinstance(result, dict) and 'probs' in result:
                return self._normalize_probs(result['probs'])
            elif hasattr(result, 'score'):
                # Convert single score to probabilities
                score = result.score
                label = result.label if hasattr(result, 'label') else 'Neutral'
                return self._label_score_to_probs(label, score)

        if self.has_predict_proba:
            probs = self.engine.predict_proba(text)
            if isinstance(probs, dict):
                return self._normalize_probs(probs)
            elif isinstance(probs, (list, tuple)):
                # Assume [neg, neu, pos] order
                return {
                    'Negative': float(probs[0]),
                    'Neutral': float(probs[1]) if len(probs) > 2 else 0.0,
                    'Positive': float(probs[-1]),
                }

        if self.has_predict:
            label = self.engine.predict(text)
            return self._label_to_probs(label)

        return {'Positive': 0.33, 'Neutral': 0.34, 'Negative': 0.33}

    def _normalize_probs(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Normalize probability keys to standard format."""
        normalized = {'Positive': 0.0, 'Neutral': 0.0, 'Negative': 0.0}

        key_map = {
            'positive': 'Positive', 'pos': 'Positive', 'Positive': 'Positive',
            'negative': 'Negative', 'neg': 'Negative', 'Negative': 'Negative',
            'neutral': 'Neutral', 'neu': 'Neutral', 'Neutral': 'Neutral',
        }

        for key, value in probs.items():
            norm_key = key_map.get(key, key_map.get(str(key).lower()))
            if norm_key:
                normalized[norm_key] = float(value)

        # Ensure they sum to 1
        total = sum(normalized.values())
        if total > 0:
            normalized = {k: v / total for k, v in normalized.items()}

        return normalized

    def _label_to_probs(self, label: str) -> Dict[str, float]:
        """Convert a label to probability distribution."""
        probs = {'Positive': 0.1, 'Neutral': 0.1, 'Negative': 0.1}
        label_norm = label.strip().lower()

        if label_norm in ['positive', 'pos']:
            probs['Positive'] = 0.8
        elif label_norm in ['negative', 'neg']:
            probs['Negative'] = 0.8
        else:
            probs['Neutral'] = 0.8

        return probs

    def _label_score_to_probs(self, label: str, score: float) -> Dict[str, float]:
        """Convert label and confidence score to probabilities."""
        # Score represents confidence in the label
        score = max(0.0, min(1.0, abs(score)))
        remaining = 1.0 - score

        label_norm = label.strip().lower() if label else 'neutral'

        probs = {
            'Positive': remaining / 2,
            'Neutral': remaining / 2,
            'Negative': remaining / 2,
        }

        if label_norm in ['positive', 'pos']:
            probs['Positive'] = score
        elif label_norm in ['negative', 'neg']:
            probs['Negative'] = score
        else:
            probs['Neutral'] = score

        return probs


class FuzzySentimentEngine:
    """
    Fuzzy Sentiment Engine - Drop-in replacement for existing engines.

    This engine combines multiple base sentiment engines using fuzzy
    inference to provide uncertainty-aware sentiment classification.

    Compatible with the existing YouTube sentiment analysis pipeline.

    Parameters
    ----------
    base_engines : dict
        Dictionary mapping engine names to engine instances
    mf_type : str
        Membership function type: 'triangular', 'gaussian', 'trapezoidal'
    defuzz_method : str
        Defuzzification method: 'centroid', 'bisector', 'mom'
    confidence_threshold : float
        Threshold below which to flag predictions as uncertain

    Example
    -------
    >>> from app.sentiment_engines import LogRegSentimentEngine, SVMSentimentEngine
    >>>
    >>> fuzzy_engine = FuzzySentimentEngine(
    ...     base_engines={
    ...         'logreg': LogRegSentimentEngine(),
    ...         'svm': SVMSentimentEngine(),
    ...     },
    ...     mf_type='gaussian',
    ...     defuzz_method='centroid'
    ... )
    >>>
    >>> result = fuzzy_engine.analyze("Great video!")
    >>> print(f"Sentiment: {result.label}, Confidence: {result.confidence:.2f}")
    """

    def __init__(
        self,
        base_engines: Dict[str, Any],
        mf_type: str = 'gaussian',
        defuzz_method: str = 'centroid',
        t_norm: str = 'min',
        t_conorm: str = 'max',
        alpha_cut: float = 0.0,
        resolution: int = 100,
        confidence_threshold: float = 0.6,
        enable_logging: bool = False
    ):
        """Initialize the Fuzzy Sentiment Engine."""
        self.base_engines = base_engines
        self.adapters = {
            name: EngineAdapter(engine, name)
            for name, engine in base_engines.items()
        }
        self.mf_type = mf_type
        self.defuzz_method = defuzz_method
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.alpha_cut = alpha_cut
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold
        self.enable_logging = enable_logging

        # Initialize fuzzy classifier
        self.classifier = FuzzySentimentClassifier(
            base_models=list(base_engines.keys()),
            mf_type=mf_type,
            defuzz_method=defuzz_method,
            t_norm=t_norm,
            t_conorm=t_conorm,
            alpha_cut=alpha_cut,
            resolution=resolution,
        )

        # Performance tracking
        self._inference_times: List[float] = []

    @classmethod
    def from_existing_engines(
        cls,
        engine_configs: List[Dict[str, str]],
        **kwargs
    ) -> 'FuzzySentimentEngine':
        """
        Create FuzzySentimentEngine from engine configuration.

        Parameters
        ----------
        engine_configs : list
            List of dicts with 'name' and 'class' keys
            Example: [{'name': 'logreg', 'class': 'LogRegSentimentEngine'}]
        **kwargs
            Additional arguments for FuzzySentimentEngine

        Returns
        -------
        FuzzySentimentEngine
            Initialized engine
        """
        # Import sentiment engines dynamically
        try:
            from app.sentiment_engines import (
                TFIDFSentimentEngine,
                LogRegSentimentEngine,
                SVMSentimentEngine,
            )
            engine_classes = {
                'TFIDFSentimentEngine': TFIDFSentimentEngine,
                'LogRegSentimentEngine': LogRegSentimentEngine,
                'SVMSentimentEngine': SVMSentimentEngine,
            }
        except ImportError:
            engine_classes = {}
            logger.warning("Could not import sentiment engines from app")

        base_engines = {}
        for config in engine_configs:
            name = config['name']
            class_name = config['class']

            if class_name in engine_classes:
                try:
                    base_engines[name] = engine_classes[class_name]()
                except Exception as e:
                    logger.warning(f"Failed to initialize {class_name}: {e}")
            else:
                logger.warning(f"Unknown engine class: {class_name}")

        if not base_engines:
            raise ValueError("No engines could be initialized")

        return cls(base_engines=base_engines, **kwargs)

    def analyze(self, text: str) -> FuzzyAnalysisResult:
        """
        Analyze text sentiment using fuzzy inference.

        This method is compatible with the existing sentiment engine API.

        Parameters
        ----------
        text : str
            Input text to classify

        Returns
        -------
        FuzzyAnalysisResult
            Classification result with uncertainty information
        """
        start_time = time.time()

        # Get probabilities from all base engines
        model_outputs = {}
        for name, adapter in self.adapters.items():
            try:
                probs = adapter.get_probabilities(text)
                model_outputs[name] = probs
            except Exception as e:
                logger.error(f"Engine '{name}' failed: {e}")
                # Use neutral fallback
                model_outputs[name] = {
                    'Positive': 0.33, 'Neutral': 0.34, 'Negative': 0.33
                }

        # Run fuzzy classification
        fuzzy_result = self.classifier.classify(model_outputs)

        # Convert to compatible result format
        result = FuzzyAnalysisResult.from_fuzzy_result(
            fuzzy_result,
            model_name=f'fuzzy_{self.defuzz_method}'
        )

        # Track performance
        inference_time = time.time() - start_time
        self._inference_times.append(inference_time)

        if self.enable_logging:
            logger.info(
                f"Fuzzy analysis: {result.label} "
                f"(score={result.score:.3f}, confidence={result.confidence:.3f}, "
                f"time={inference_time*1000:.1f}ms)"
            )

        return result

    def analyze_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[FuzzyAnalysisResult]:
        """
        Analyze multiple texts.

        Parameters
        ----------
        texts : list
            List of texts to classify
        show_progress : bool
            Whether to show progress indicator

        Returns
        -------
        list
            List of FuzzyAnalysisResult objects
        """
        results = []

        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(texts)} samples")

            results.append(self.analyze(text))

        return results

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self._inference_times:
            return {}

        import numpy as np
        times = np.array(self._inference_times)

        return {
            'mean_inference_time_ms': float(np.mean(times) * 1000),
            'std_inference_time_ms': float(np.std(times) * 1000),
            'min_inference_time_ms': float(np.min(times) * 1000),
            'max_inference_time_ms': float(np.max(times) * 1000),
            'total_samples': len(times),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fuzzy model configuration."""
        return {
            'base_models': list(self.base_engines.keys()),
            'mf_type': self.mf_type,
            'defuzz_method': self.defuzz_method,
            't_norm': self.t_norm,
            't_conorm': self.t_conorm,
            'alpha_cut': self.alpha_cut,
            'resolution': self.resolution,
            'confidence_threshold': self.confidence_threshold,
            'num_rules': len(self.classifier.fis.rules),
        }


def create_default_fuzzy_engine() -> FuzzySentimentEngine:
    """
    Create a default FuzzySentimentEngine with standard configuration.

    Uses LogReg, SVM, and TF-IDF as base models with Gaussian
    membership functions and centroid defuzzification.

    Returns
    -------
    FuzzySentimentEngine
        Configured fuzzy engine
    """
    return FuzzySentimentEngine.from_existing_engines(
        engine_configs=[
            {'name': 'logreg', 'class': 'LogRegSentimentEngine'},
            {'name': 'svm', 'class': 'SVMSentimentEngine'},
            {'name': 'tfidf', 'class': 'TFIDFSentimentEngine'},
        ],
        mf_type='gaussian',
        defuzz_method='centroid',
        confidence_threshold=0.6
    )
