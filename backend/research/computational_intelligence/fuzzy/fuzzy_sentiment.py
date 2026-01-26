"""
Fuzzy Sentiment Classifier - Main Integration Layer

This module provides the primary interface for fuzzy sentiment classification,
integrating fuzzy inference with existing ML sentiment engines.

Key Innovation (Thesis Contribution):
    Traditional sentiment classifiers output crisp labels (Positive/Negative/Neutral)
    with probability scores. However, they cannot express:
    - "Mostly positive but slightly uncertain"
    - "Between neutral and negative"
    - "High confidence negative"

    The FuzzySentimentClassifier addresses this by:
    1. Accepting scores from multiple base models
    2. Fuzzifying these scores into linguistic variables
    3. Applying fuzzy rules to handle model disagreement
    4. Producing sentiment with explicit uncertainty quantification

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    FuzzySentimentClassifier                  │
    ├─────────────────────────────────────────────────────────────┤
    │  Input: ML Model Probabilities                              │
    │    [LogReg: 0.72, SVM: 0.68, BERT: 0.81]                   │
    │                           ↓                                 │
    │  Fuzzification: Convert to fuzzy membership                 │
    │    LogReg: {neg: 0.0, neu: 0.1, pos: 0.9}                  │
    │    SVM:    {neg: 0.0, neu: 0.2, pos: 0.8}                  │
    │    BERT:   {neg: 0.0, neu: 0.0, pos: 1.0}                  │
    │                           ↓                                 │
    │  Fuzzy Inference: Apply rules                               │
    │    R1: IF all positive THEN positive (w=1.0)               │
    │    R2: IF mixed THEN uncertain (w=0.7)                     │
    │                           ↓                                 │
    │  Defuzzification: Convert to crisp output                   │
    │                           ↓                                 │
    │  Output: Sentiment + Uncertainty Metrics                    │
    │    {label: 'Positive', score: 0.78,                        │
    │     uncertainty: 0.12, confidence: 0.88}                   │
    └─────────────────────────────────────────────────────────────┘

Reference:
    This approach is inspired by:
    - Zadeh (1965): Fuzzy Sets
    - Mamdani (1974): Fuzzy Inference Systems
    - Cambria et al. (2020): Fuzzy Sentiment Analysis

Author: [Your Name]
Thesis: Computational Intelligence Approaches for YouTube Sentiment Analysis
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import json
from pathlib import Path

# NumPy 2.0+ compatibility: trapz was renamed to trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

from .membership_functions import (
    MembershipFunction,
    TriangularMF,
    TrapezoidalMF,
    GaussianMF,
    create_three_class_mfs,
    create_sentiment_mfs_triangular,
    create_sentiment_mfs_gaussian,
)
from .fuzzy_inference import (
    FuzzyVariable,
    FuzzyRule,
    FuzzyInferenceSystem,
    create_sentiment_fis,
    FuzzyOperator,
)
from .defuzzification import (
    Defuzzifier,
    DefuzzMethod,
    compute_uncertainty_metrics,
    compare_defuzzification_methods,
)


@dataclass
class FuzzySentimentResult:
    """
    Complete result from fuzzy sentiment classification.

    This dataclass captures both the classification output and
    the uncertainty information crucial for thesis analysis.

    Attributes:
        label: Final sentiment label ('Positive', 'Neutral', 'Negative')
        crisp_score: Defuzzified sentiment score [0, 1]
        probabilities: Class probabilities derived from fuzzy output
        uncertainty_metrics: Dictionary of uncertainty measures
        fuzzified_inputs: Raw fuzzified values from each input model
        rule_activations: Which rules fired and their strengths
        defuzz_comparison: Output from different defuzzification methods
        metadata: Additional information (model names, parameters, etc.)
    """
    label: str
    crisp_score: float
    probabilities: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    fuzzified_inputs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rule_activations: List[Tuple[str, float]] = field(default_factory=list)
    defuzz_comparison: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'label': self.label,
            'crisp_score': self.crisp_score,
            'probabilities': self.probabilities,
            'uncertainty_metrics': self.uncertainty_metrics,
            'fuzzified_inputs': self.fuzzified_inputs,
            'rule_activations': self.rule_activations,
            'defuzz_comparison': self.defuzz_comparison,
            'metadata': self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def confidence(self) -> float:
        """Overall confidence in the classification."""
        return 1.0 - self.uncertainty_metrics.get('fuzziness', 0.0)

    @property
    def is_uncertain(self) -> bool:
        """Check if classification has high uncertainty."""
        return self.uncertainty_metrics.get('fuzziness', 0) > 0.3


class FuzzySentimentClassifier:
    """
    Fuzzy Logic-based Sentiment Classifier with Uncertainty Quantification.

    This classifier wraps existing ML sentiment models and adds a fuzzy
    inference layer to handle uncertainty and model disagreement.

    Key Features:
        1. Multi-model ensemble integration
        2. Explicit uncertainty quantification
        3. Configurable membership functions
        4. Multiple defuzzification strategies
        5. Rule-based conflict resolution

    Parameters
    ----------
    base_models : list, optional
        Names of base models to use (default: ['logreg', 'svm', 'tfidf'])
    mf_type : str
        Membership function type: 'triangular', 'gaussian', 'trapezoidal'
    defuzz_method : str
        Defuzzification method: 'centroid', 'bisector', 'mom', 'som', 'lom'
    t_norm : str
        T-norm for fuzzy AND: 'min', 'product', 'lukasiewicz'
    alpha_cut : float
        Threshold for alpha-cut operations [0, 1]
    resolution : int
        Number of points for fuzzy set discretization

    Example
    -------
    >>> classifier = FuzzySentimentClassifier(
    ...     base_models=['logreg', 'svm', 'bert'],
    ...     mf_type='gaussian',
    ...     defuzz_method='centroid'
    ... )
    >>>
    >>> # Get scores from base models
    >>> model_scores = {
    ...     'logreg': {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1},
    ...     'svm': {'positive': 0.8, 'neutral': 0.15, 'negative': 0.05},
    ...     'bert': {'positive': 0.85, 'neutral': 0.1, 'negative': 0.05}
    ... }
    >>>
    >>> result = classifier.classify(model_scores)
    >>> print(f"Sentiment: {result.label} (confidence: {result.confidence:.2f})")
    """

    # Standard sentiment class boundaries
    CLASS_BOUNDARIES = [
        (0.33, 'Negative'),
        (0.66, 'Neutral'),
        (1.0, 'Positive')
    ]

    def __init__(
        self,
        base_models: Optional[List[str]] = None,
        mf_type: str = 'gaussian',
        defuzz_method: str = 'centroid',
        t_norm: str = 'min',
        t_conorm: str = 'max',
        alpha_cut: float = 0.0,
        resolution: int = 100,
        custom_rules: Optional[List[FuzzyRule]] = None
    ):
        """Initialize the Fuzzy Sentiment Classifier."""
        self.base_models = base_models or ['logreg', 'svm', 'tfidf']
        self.mf_type = mf_type
        self.defuzz_method = defuzz_method
        self.t_norm = FuzzyOperator(t_norm)
        self.t_conorm = FuzzyOperator(t_conorm)
        self.alpha_cut = alpha_cut
        self.resolution = resolution

        # Initialize components
        self.defuzzifier = Defuzzifier(method=defuzz_method, resolution=resolution)
        self.fis = self._build_inference_system(custom_rules)

        # Universe of discourse for sentiment scores
        self.universe = np.linspace(0, 1, resolution)

    def _build_inference_system(
        self,
        custom_rules: Optional[List[FuzzyRule]] = None
    ) -> FuzzyInferenceSystem:
        """
        Build the fuzzy inference system.

        Creates input variables for each base model and output variable
        for the final sentiment, with appropriate fuzzy rules.
        """
        fis = FuzzyInferenceSystem(
            t_norm=self.t_norm,
            t_conorm=self.t_conorm,
            aggregation='max',
            implication='min'
        )

        # Create input variables for each model
        for model_name in self.base_models:
            mfs = create_three_class_mfs(mf_type=self.mf_type)
            input_var = FuzzyVariable(
                name=model_name,
                universe=(0.0, 1.0),
                mfs=mfs,
                resolution=self.resolution
            )
            fis.add_input_variable(input_var)

        # Create output variable
        output_mfs = create_three_class_mfs(mf_type=self.mf_type)
        output_var = FuzzyVariable(
            name='sentiment',
            universe=(0.0, 1.0),
            mfs=output_mfs,
            resolution=self.resolution
        )
        fis.add_output_variable(output_var)

        # Add rules
        if custom_rules:
            fis.add_rules(custom_rules)
        else:
            self._add_default_rules(fis)

        return fis

    def _add_default_rules(self, fis: FuzzyInferenceSystem) -> None:
        """
        Add default fuzzy rules for sentiment classification.

        Rule Design Philosophy:
        - Agreement rules (all models agree): High weight
        - Majority rules (most models agree): Medium weight
        - Conflict resolution rules: Lower weight, tend toward neutral
        """
        sentiments = ['negative', 'neutral', 'positive']
        n_models = len(self.base_models)

        # Rule 1: All models agree (strongest rules)
        for sent in sentiments:
            fis.add_rule(FuzzyRule(
                antecedents=[(model, sent) for model in self.base_models],
                consequent=('sentiment', sent),
                weight=1.0,
                operator='AND'
            ))

        # Rule 2: Pairwise agreement rules (for 2+ models)
        if n_models >= 2:
            for i, model1 in enumerate(self.base_models):
                for model2 in self.base_models[i+1:]:
                    for sent in sentiments:
                        fis.add_rule(FuzzyRule(
                            antecedents=[(model1, sent), (model2, sent)],
                            consequent=('sentiment', sent),
                            weight=0.8,
                            operator='AND'
                        ))

        # Rule 3: Conflict resolution rules
        if n_models >= 2:
            m1, m2 = self.base_models[0], self.base_models[1]

            # Positive vs Negative → Neutral
            fis.add_rule(FuzzyRule(
                antecedents=[(m1, 'positive'), (m2, 'negative')],
                consequent=('sentiment', 'neutral'),
                weight=0.6,
                operator='AND'
            ))
            fis.add_rule(FuzzyRule(
                antecedents=[(m1, 'negative'), (m2, 'positive')],
                consequent=('sentiment', 'neutral'),
                weight=0.6,
                operator='AND'
            ))

            # Positive vs Neutral → Slight positive
            fis.add_rule(FuzzyRule(
                antecedents=[(m1, 'positive'), (m2, 'neutral')],
                consequent=('sentiment', 'positive'),
                weight=0.5,
                operator='AND'
            ))
            fis.add_rule(FuzzyRule(
                antecedents=[(m1, 'neutral'), (m2, 'positive')],
                consequent=('sentiment', 'positive'),
                weight=0.5,
                operator='AND'
            ))

            # Negative vs Neutral → Slight negative
            fis.add_rule(FuzzyRule(
                antecedents=[(m1, 'negative'), (m2, 'neutral')],
                consequent=('sentiment', 'negative'),
                weight=0.5,
                operator='AND'
            ))
            fis.add_rule(FuzzyRule(
                antecedents=[(m1, 'neutral'), (m2, 'negative')],
                consequent=('sentiment', 'negative'),
                weight=0.5,
                operator='AND'
            ))

        # Rule 4: Single model fallback rules (lowest priority)
        for model in self.base_models:
            for sent in sentiments:
                fis.add_rule(FuzzyRule(
                    antecedents=[(model, sent)],
                    consequent=('sentiment', sent),
                    weight=0.3,
                    operator='AND'
                ))

    def _convert_to_positive_score(
        self,
        model_output: Union[Dict[str, float], float, np.ndarray]
    ) -> float:
        """
        Convert various model output formats to a positive sentiment score.

        Handles:
        - Dictionary: {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1}
        - Float: Direct score in [0, 1]
        - Array: [neg_prob, neu_prob, pos_prob]

        Returns a score where 1.0 = fully positive, 0.0 = fully negative.
        """
        if isinstance(model_output, dict):
            # Convert class probabilities to continuous score
            pos = model_output.get('positive', model_output.get('Positive', 0))
            neg = model_output.get('negative', model_output.get('Negative', 0))
            neu = model_output.get('neutral', model_output.get('Neutral', 0))

            # Weighted score: positive pushes toward 1, negative toward 0
            # Neutral contributes to middle
            return pos * 1.0 + neu * 0.5 + neg * 0.0

        elif isinstance(model_output, (list, np.ndarray)):
            arr = np.array(model_output)
            if len(arr) == 3:
                # Assume [negative, neutral, positive]
                return arr[2] * 1.0 + arr[1] * 0.5 + arr[0] * 0.0
            elif len(arr) == 2:
                # Assume [negative, positive]
                return arr[1]
            else:
                return float(arr[0])

        else:
            return float(model_output)

    def classify(
        self,
        model_outputs: Dict[str, Union[Dict[str, float], float]],
        return_details: bool = True
    ) -> FuzzySentimentResult:
        """
        Perform fuzzy sentiment classification.

        Parameters
        ----------
        model_outputs : dict
            Dictionary mapping model names to their outputs.
            Each output can be:
            - Dict: {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1}
            - Float: Positive sentiment score [0, 1]
        return_details : bool
            Whether to compute detailed metrics

        Returns
        -------
        FuzzySentimentResult
            Complete classification result with uncertainty metrics

        Example
        -------
        >>> outputs = {
        ...     'logreg': {'positive': 0.8, 'neutral': 0.15, 'negative': 0.05},
        ...     'svm': {'positive': 0.75, 'neutral': 0.2, 'negative': 0.05}
        ... }
        >>> result = classifier.classify(outputs)
        """
        # Convert model outputs to positive scores for FIS input
        crisp_inputs = {}
        for model_name in self.base_models:
            if model_name not in model_outputs:
                raise KeyError(f"Model '{model_name}' output not provided")
            crisp_inputs[model_name] = self._convert_to_positive_score(
                model_outputs[model_name]
            )

        # Run fuzzy inference
        aggregated, details = self.fis.evaluate(crisp_inputs, return_details=True)

        # Get aggregated output for sentiment
        output_membership = aggregated['sentiment']
        output_var = self.fis.output_variables['sentiment']
        universe = output_var.get_universe_array()

        # Defuzzify
        crisp_score = self.defuzzifier.defuzzify(universe, output_membership)

        # Determine class label
        label = self._score_to_label(crisp_score)

        # Compute probabilities from fuzzy output
        probabilities = self._compute_class_probabilities(universe, output_membership)

        # Compute uncertainty metrics
        uncertainty = compute_uncertainty_metrics(universe, output_membership)

        # Compare defuzzification methods (for thesis analysis)
        defuzz_comparison = {}
        if return_details:
            defuzz_comparison = compare_defuzzification_methods(
                universe, output_membership
            )

        return FuzzySentimentResult(
            label=label,
            crisp_score=crisp_score,
            probabilities=probabilities,
            uncertainty_metrics=uncertainty,
            fuzzified_inputs=details['fuzzified_inputs'],
            rule_activations=details['rule_evaluations'],
            defuzz_comparison=defuzz_comparison,
            metadata={
                'base_models': self.base_models,
                'mf_type': self.mf_type,
                'defuzz_method': self.defuzz_method,
                'input_scores': crisp_inputs,
            }
        )

    def classify_batch(
        self,
        batch_outputs: List[Dict[str, Union[Dict[str, float], float]]],
        return_details: bool = False
    ) -> List[FuzzySentimentResult]:
        """
        Classify multiple samples efficiently.

        Parameters
        ----------
        batch_outputs : list
            List of model output dictionaries
        return_details : bool
            Whether to compute detailed metrics for each sample

        Returns
        -------
        list
            List of FuzzySentimentResult objects
        """
        return [
            self.classify(outputs, return_details=return_details)
            for outputs in batch_outputs
        ]

    def _score_to_label(self, score: float) -> str:
        """Convert crisp score to sentiment label."""
        for threshold, label in self.CLASS_BOUNDARIES:
            if score <= threshold:
                return label
        return self.CLASS_BOUNDARIES[-1][1]

    def _compute_class_probabilities(
        self,
        universe: np.ndarray,
        membership: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute pseudo-probabilities for each sentiment class.

        Uses the area under each class's membership function region
        as a proxy for class probability.
        """
        output_mfs = self.fis.output_variables['sentiment'].mfs

        class_areas = {}
        for class_name, mf in output_mfs.items():
            # Compute intersection of aggregated output with class MF
            class_membership = mf(universe)
            intersection = np.minimum(membership, class_membership)
            area = _trapz(intersection, universe)
            class_areas[class_name] = area

        # Normalize to get probabilities
        total_area = sum(class_areas.values())
        if total_area > 0:
            probabilities = {
                name.capitalize(): area / total_area
                for name, area in class_areas.items()
            }
        else:
            probabilities = {name.capitalize(): 1/3 for name in class_areas}

        return probabilities

    def get_rules_summary(self) -> str:
        """Get a formatted summary of fuzzy rules."""
        return self.fis.get_rule_summary()

    def visualize_membership_functions(self) -> Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Get data for visualizing membership functions.

        Returns
        -------
        dict
            Dictionary with 'input' and 'output' keys, each containing
            (universe, {set_name: membership_values}) tuples
        """
        result = {'input': {}, 'output': {}}

        for var_name, var in self.fis.input_variables.items():
            universe = var.get_universe_array()
            mf_values = {
                name: mf(universe)
                for name, mf in var.mfs.items()
            }
            result['input'][var_name] = (universe, mf_values)

        for var_name, var in self.fis.output_variables.items():
            universe = var.get_universe_array()
            mf_values = {
                name: mf(universe)
                for name, mf in var.mfs.items()
            }
            result['output'][var_name] = (universe, mf_values)

        return result


class FuzzyEnsembleSentimentEngine:
    """
    Integration layer between existing sentiment engines and fuzzy classification.

    This class wraps existing sentiment engines (LogReg, SVM, BERT, etc.)
    and applies fuzzy inference for uncertainty-aware classification.

    Usage with existing system:
        >>> from app.sentiment_engines import LogRegSentimentEngine, SVMSentimentEngine
        >>> from computational_intelligence.fuzzy import FuzzyEnsembleSentimentEngine
        >>>
        >>> fuzzy_engine = FuzzyEnsembleSentimentEngine(
        ...     engines={
        ...         'logreg': LogRegSentimentEngine(),
        ...         'svm': SVMSentimentEngine(),
        ...     }
        ... )
        >>>
        >>> result = fuzzy_engine.predict("This video is amazing!")
    """

    def __init__(
        self,
        engines: Optional[Dict[str, Any]] = None,
        fuzzy_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the fuzzy ensemble engine.

        Parameters
        ----------
        engines : dict, optional
            Dictionary mapping names to sentiment engine instances
        fuzzy_config : dict, optional
            Configuration for FuzzySentimentClassifier
        """
        self.engines = engines or {}
        fuzzy_config = fuzzy_config or {}

        # Initialize fuzzy classifier
        self.fuzzy_classifier = FuzzySentimentClassifier(
            base_models=list(self.engines.keys()),
            **fuzzy_config
        )

    def add_engine(self, name: str, engine: Any) -> 'FuzzyEnsembleSentimentEngine':
        """Add a sentiment engine to the ensemble."""
        self.engines[name] = engine
        # Rebuild fuzzy classifier with new model
        self.fuzzy_classifier = FuzzySentimentClassifier(
            base_models=list(self.engines.keys()),
            mf_type=self.fuzzy_classifier.mf_type,
            defuzz_method=self.fuzzy_classifier.defuzz_method,
        )
        return self

    def predict(self, text: str) -> FuzzySentimentResult:
        """
        Predict sentiment for a single text.

        Parameters
        ----------
        text : str
            Input text to classify

        Returns
        -------
        FuzzySentimentResult
            Fuzzy classification result
        """
        # Get predictions from all base engines
        model_outputs = {}
        for name, engine in self.engines.items():
            # Assume engine has predict_proba or similar method
            if hasattr(engine, 'predict_proba'):
                probs = engine.predict_proba(text)
                if isinstance(probs, dict):
                    model_outputs[name] = probs
                else:
                    # Assume [neg, neu, pos] order
                    model_outputs[name] = {
                        'negative': probs[0],
                        'neutral': probs[1] if len(probs) > 2 else 0,
                        'positive': probs[-1]
                    }
            elif hasattr(engine, 'analyze'):
                result = engine.analyze(text)
                model_outputs[name] = result.get('probabilities', result)
            else:
                raise AttributeError(
                    f"Engine '{name}' must have 'predict_proba' or 'analyze' method"
                )

        return self.fuzzy_classifier.classify(model_outputs)

    def predict_batch(self, texts: List[str]) -> List[FuzzySentimentResult]:
        """Predict sentiment for multiple texts."""
        return [self.predict(text) for text in texts]
