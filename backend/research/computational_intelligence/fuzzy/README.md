# Fuzzy Sentiment Classification Module

## Thesis-Grade Implementation for Computational Intelligence

This module implements a **Fuzzy Logic-based Sentiment Classification System** as part of a Master's thesis in Computational Intelligence. It provides uncertainty-aware sentiment analysis that extends beyond traditional machine learning approaches.

---

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Thesis Experiments](#thesis-experiments)
7. [API Reference](#api-reference)
8. [References](#references)

---

## Theoretical Foundation

### Why Fuzzy Logic for Sentiment Analysis?

Traditional sentiment classifiers output crisp labels (Positive/Negative/Neutral) with probability scores. However, they cannot adequately express:

- **Uncertainty**: "I'm 70% confident this is positive" vs "This is definitely positive"
- **Ambiguity**: Comments that genuinely lie between sentiment classes
- **Model Disagreement**: When different models produce conflicting predictions

**Fuzzy Logic** addresses these limitations through:

1. **Fuzzy Sets**: Allow partial membership in sentiment categories
2. **Linguistic Variables**: Express sentiment in human-interpretable terms
3. **Fuzzy Rules**: Encode expert knowledge for handling edge cases
4. **Uncertainty Quantification**: Explicit measures of classification confidence

### Mathematical Foundation

**Fuzzy Set Definition** (Zadeh, 1965):
```
A fuzzy set A in universe X is defined by membership function μ_A: X → [0,1]
where μ_A(x) represents the degree to which x belongs to A.
```

**Fuzzy Inference Process**:
```
1. Fuzzification: Crisp inputs → Fuzzy membership degrees
2. Rule Evaluation: Apply IF-THEN rules with fuzzy operators
3. Aggregation: Combine rule outputs
4. Defuzzification: Fuzzy output → Crisp value
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FuzzySentimentClassifier                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │   LogReg     │   │     SVM      │   │    BERT      │        │
│  │   Engine     │   │    Engine    │   │   Engine     │        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                  │                 │
│         └────────────┬─────┴─────────────────┘                 │
│                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   FUZZIFICATION                          │   │
│  │   • Triangular/Gaussian/Trapezoidal MFs                 │   │
│  │   • Convert probabilities to fuzzy memberships          │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  FUZZY INFERENCE                         │   │
│  │   • Mamdani-type inference                              │   │
│  │   • Agreement/Disagreement rules                         │   │
│  │   • T-norms (min, product) / T-conorms (max, sum)       │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  DEFUZZIFICATION                         │   │
│  │   • Centroid (CoG)                                      │   │
│  │   • Bisector                                            │   │
│  │   • Mean/Smallest/Largest of Maximum                    │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              OUTPUT + UNCERTAINTY                        │   │
│  │   • Sentiment label & crisp score                       │   │
│  │   • Fuzziness index                                     │   │
│  │   • Confidence measure                                  │   │
│  │   • Class probabilities                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Membership Functions (`membership_functions.py`)

| Type | Description | Best For |
|------|-------------|----------|
| Triangular | Three-point linear | Simple, fast computation |
| Trapezoidal | Four-point with plateau | Definite class regions |
| Gaussian | Smooth bell curve | Neural network integration |
| Sigmoid | S-shaped asymmetric | Boundary classes |
| Generalized Bell | Adjustable shoulders | Fine-tuning |

### 2. Fuzzy Inference System (`fuzzy_inference.py`)

- **Fuzzy Variables**: Input/output linguistic variables
- **Fuzzy Rules**: IF-THEN rules with weights
- **Operators**: T-norms (AND), T-conorms (OR)
- **Aggregation**: max, sum, probabilistic OR

### 3. Defuzzification (`defuzzification.py`)

| Method | Formula | Characteristics |
|--------|---------|-----------------|
| Centroid | ∫x·μ(x)dx / ∫μ(x)dx | Smooth, considers entire set |
| Bisector | Area division point | Balanced |
| MOM | mean(argmax μ) | Focuses on certainty |
| SOM/LOM | min/max(argmax μ) | Extreme values |

### 4. Main Classifier (`fuzzy_sentiment.py`)

```python
class FuzzySentimentClassifier:
    """
    Main fuzzy sentiment classifier with uncertainty quantification.
    """
```

### 5. Evaluation Metrics (`fuzzy_evaluation.py`)

**Standard Metrics**: Accuracy, Precision, Recall, F1

**Fuzzy-Specific Metrics**:
- Fuzziness Index: Average uncertainty
- Specificity: Inverse of spread
- Uncertainty Discrimination: Difference in uncertainty for correct/incorrect predictions
- Selective Accuracy: Accuracy when confident

**Calibration Metrics**:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score

---

## Installation

The fuzzy module is part of the research package. No additional installation required.

```python
# From project root
from research.computational_intelligence.fuzzy import (
    FuzzySentimentClassifier,
    FuzzyEvaluator,
)
```

### Dependencies
- numpy
- (Optional) matplotlib for visualization

---

## Usage

### Basic Usage

```python
from research.computational_intelligence.fuzzy import FuzzySentimentClassifier

# Initialize classifier
classifier = FuzzySentimentClassifier(
    base_models=['logreg', 'svm', 'tfidf'],
    mf_type='gaussian',
    defuzz_method='centroid'
)

# Classify with model outputs
model_outputs = {
    'logreg': {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1},
    'svm': {'positive': 0.8, 'neutral': 0.15, 'negative': 0.05},
    'tfidf': {'positive': 0.65, 'neutral': 0.25, 'negative': 0.1},
}

result = classifier.classify(model_outputs)

print(f"Label: {result.label}")
print(f"Score: {result.crisp_score:.4f}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Fuzziness: {result.uncertainty_metrics['fuzziness']:.4f}")
```

### Integration with Existing Engines

```python
from app.sentiment_engines import LogRegSentimentEngine, SVMSentimentEngine
from research.computational_intelligence.fuzzy import FuzzySentimentEngine

# Create fuzzy engine with existing engines
fuzzy_engine = FuzzySentimentEngine(
    base_engines={
        'logreg': LogRegSentimentEngine(),
        'svm': SVMSentimentEngine(),
    },
    mf_type='gaussian',
    defuzz_method='centroid'
)

# Use like any other engine
result = fuzzy_engine.analyze("This video is amazing!")
print(f"Sentiment: {result.label} (confidence: {result.confidence:.2f})")
```

### Evaluation

```python
from research.computational_intelligence.fuzzy import FuzzyEvaluator

evaluator = FuzzyEvaluator()

# Add predictions
for text, true_label in test_data:
    result = classifier.classify(get_model_outputs(text))
    evaluator.add_sample(
        true_label=true_label,
        predicted_label=result.label,
        probabilities=result.probabilities,
        uncertainty=result.uncertainty_metrics,
        crisp_score=result.crisp_score
    )

# Compute metrics
evaluation = evaluator.compute_metrics()
print(evaluation.summary())

# Generate LaTeX table for thesis
print(evaluation.to_latex_table())
```

---

## Thesis Experiments

### Experiment 1: Membership Function Comparison

Compare triangular, Gaussian, and trapezoidal MFs:

```python
mf_types = ['triangular', 'gaussian', 'trapezoidal']
results = {}

for mf_type in mf_types:
    classifier = FuzzySentimentClassifier(mf_type=mf_type)
    # Run evaluation...
    results[mf_type] = evaluation_metrics
```

### Experiment 2: Defuzzification Method Comparison

```python
defuzz_methods = ['centroid', 'bisector', 'mom', 'som', 'lom']
results = {}

for method in defuzz_methods:
    classifier = FuzzySentimentClassifier(defuzz_method=method)
    # Run evaluation...
```

### Experiment 3: Uncertainty Quality Assessment

```python
# Evaluate if uncertainty correlates with errors
evaluation = evaluator.compute_metrics()

print(f"Uncertainty when correct: {evaluation.uncertainty_metrics['uncertainty_when_correct']:.4f}")
print(f"Uncertainty when incorrect: {evaluation.uncertainty_metrics['uncertainty_when_incorrect']:.4f}")
print(f"Discrimination: {evaluation.uncertainty_metrics['uncertainty_discrimination']:.4f}")
```

### Experiment 4: Comparison with Baseline

```python
# Compare fuzzy classifier with standard ensemble
comparison = evaluator.compare_with_baseline(
    baseline_predictions=baseline_preds,
    baseline_name='Standard Ensemble'
)

print(f"Fuzzy accuracy: {comparison['fuzzy_accuracy']:.4f}")
print(f"Baseline accuracy: {comparison['standard_ensemble_accuracy']:.4f}")
print(f"McNemar's χ²: {comparison['mcnemar_chi2']:.4f}")
```

---

## API Reference

### FuzzySentimentClassifier

```python
FuzzySentimentClassifier(
    base_models: List[str] = ['logreg', 'svm', 'tfidf'],
    mf_type: str = 'gaussian',           # 'triangular', 'gaussian', 'trapezoidal'
    defuzz_method: str = 'centroid',     # 'centroid', 'bisector', 'mom', 'som', 'lom'
    t_norm: str = 'min',                 # 'min', 'product', 'lukasiewicz'
    t_conorm: str = 'max',               # 'max', 'prob_sum', 'bounded_sum'
    alpha_cut: float = 0.0,
    resolution: int = 100,
)
```

### FuzzySentimentResult

```python
@dataclass
class FuzzySentimentResult:
    label: str                              # 'Positive', 'Neutral', 'Negative'
    crisp_score: float                      # [0, 1]
    probabilities: Dict[str, float]         # Class probabilities
    uncertainty_metrics: Dict[str, float]   # Fuzziness, specificity, etc.
    fuzzified_inputs: Dict[str, Dict]       # Raw fuzzified values
    rule_activations: List[Tuple]           # Fired rules
    defuzz_comparison: Dict[str, float]     # All defuzz method outputs
    metadata: Dict[str, Any]                # Configuration info

    @property
    def confidence(self) -> float           # 1 - fuzziness
    @property
    def is_uncertain(self) -> bool          # fuzziness > 0.3
```

### FuzzyEvaluator

```python
FuzzyEvaluator(
    classes: List[str] = ['Negative', 'Neutral', 'Positive'],
    uncertainty_threshold: float = 0.3,
)

# Methods
evaluator.add_sample(true_label, predicted_label, probabilities, uncertainty, crisp_score)
evaluator.add_batch(true_labels, results)
evaluation = evaluator.compute_metrics()
comparison = evaluator.compare_with_baseline(baseline_preds, baseline_name)
```

---

## References

1. Zadeh, L.A. (1965). "Fuzzy Sets". Information and Control, 8(3), 338-353.

2. Mamdani, E.H., & Assilian, S. (1975). "An experiment in linguistic synthesis with a fuzzy logic controller". International Journal of Man-Machine Studies, 7(1), 1-13.

3. Cambria, E., et al. (2020). "SenticNet 6: Ensemble Application of Symbolic and Subsymbolic AI for Sentiment Analysis". CIKM 2020.

4. Hullermeier, E., & Waegeman, W. (2021). "Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods". Machine Learning, 110(3), 457-506.

5. Lee, C.C. (1990). "Fuzzy Logic in Control Systems: Fuzzy Logic Controller—Part I". IEEE Transactions on Systems, Man, and Cybernetics, 20(2), 404-418.

---

## Author

[Your Name]
Master's Thesis: Computational Intelligence Approaches for YouTube Sentiment Analysis

---

## License

Part of the YouTube Sentiment Analysis project.
