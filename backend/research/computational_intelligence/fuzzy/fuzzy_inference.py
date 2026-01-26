"""
Fuzzy Inference System for Sentiment Analysis

This module implements a Mamdani-type Fuzzy Inference System (FIS)
for combining multiple sentiment signals and handling uncertainty.

Architecture:
    1. Fuzzification: Convert crisp inputs to fuzzy membership degrees
    2. Rule Evaluation: Apply fuzzy rules with AND/OR operations
    3. Aggregation: Combine rule outputs
    4. Defuzzification: Convert fuzzy output to crisp value

Key Features:
    - Support for multiple input variables (multi-model ensemble)
    - Configurable fuzzy operators (min/max, product/sum)
    - Weighted rule importance
    - Type-1 and Interval Type-2 fuzzy support

Reference:
    Mamdani, E.H., & Assilian, S. (1975). "An experiment in linguistic
    synthesis with a fuzzy logic controller". International Journal of
    Man-Machine Studies, 7(1), 1-13.

Author: [Your Name]
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .membership_functions import MembershipFunction


class FuzzyOperator(Enum):
    """
    Fuzzy operators for combining membership degrees.

    AND Operators (T-norms):
        - MIN: Standard fuzzy intersection (Zadeh)
        - PRODUCT: Algebraic product (smoother)
        - LUKASIEWICZ: max(0, a + b - 1)

    OR Operators (T-conorms):
        - MAX: Standard fuzzy union (Zadeh)
        - PROBABILISTIC_SUM: a + b - a*b
        - BOUNDED_SUM: min(1, a + b)
    """
    # T-norms (AND)
    MIN = 'min'
    PRODUCT = 'product'
    LUKASIEWICZ = 'lukasiewicz'

    # T-conorms (OR)
    MAX = 'max'
    PROBABILISTIC_SUM = 'prob_sum'
    BOUNDED_SUM = 'bounded_sum'


def fuzzy_and(
    a: float,
    b: float,
    operator: FuzzyOperator = FuzzyOperator.MIN
) -> float:
    """
    Compute fuzzy AND (T-norm) of two membership degrees.

    Parameters
    ----------
    a : float
        First membership degree [0, 1]
    b : float
        Second membership degree [0, 1]
    operator : FuzzyOperator
        T-norm to use

    Returns
    -------
    float
        Combined membership degree
    """
    if operator == FuzzyOperator.MIN:
        return min(a, b)
    elif operator == FuzzyOperator.PRODUCT:
        return a * b
    elif operator == FuzzyOperator.LUKASIEWICZ:
        return max(0, a + b - 1)
    else:
        raise ValueError(f"Invalid AND operator: {operator}")


def fuzzy_or(
    a: float,
    b: float,
    operator: FuzzyOperator = FuzzyOperator.MAX
) -> float:
    """
    Compute fuzzy OR (T-conorm) of two membership degrees.

    Parameters
    ----------
    a : float
        First membership degree [0, 1]
    b : float
        Second membership degree [0, 1]
    operator : FuzzyOperator
        T-conorm to use

    Returns
    -------
    float
        Combined membership degree
    """
    if operator == FuzzyOperator.MAX:
        return max(a, b)
    elif operator == FuzzyOperator.PROBABILISTIC_SUM:
        return a + b - a * b
    elif operator == FuzzyOperator.BOUNDED_SUM:
        return min(1, a + b)
    else:
        raise ValueError(f"Invalid OR operator: {operator}")


def fuzzy_not(a: float) -> float:
    """
    Compute fuzzy NOT (complement) of a membership degree.

    Parameters
    ----------
    a : float
        Membership degree [0, 1]

    Returns
    -------
    float
        Complement (1 - a)
    """
    return 1.0 - a


@dataclass
class FuzzyVariable:
    """
    A fuzzy linguistic variable with associated membership functions.

    A fuzzy variable represents a concept (e.g., "sentiment_score") with
    multiple fuzzy sets (e.g., "negative", "neutral", "positive").

    Attributes:
        name: Variable identifier
        universe: Range of possible values (min, max)
        mfs: Dictionary of membership functions for each fuzzy set
        resolution: Number of points for discretization

    Example:
        >>> sentiment_var = FuzzyVariable(
        ...     name='sentiment',
        ...     universe=(0, 1),
        ...     mfs={
        ...         'negative': TriangularMF('neg', 0, 0, 0.5),
        ...         'neutral': TriangularMF('neu', 0.25, 0.5, 0.75),
        ...         'positive': TriangularMF('pos', 0.5, 1, 1),
        ...     }
        ... )
    """
    name: str
    universe: Tuple[float, float]
    mfs: Dict[str, MembershipFunction]
    resolution: int = 100

    def fuzzify(self, value: float) -> Dict[str, float]:
        """
        Convert a crisp value to fuzzy membership degrees.

        Parameters
        ----------
        value : float
            Crisp input value

        Returns
        -------
        dict
            Mapping from fuzzy set names to membership degrees
        """
        return {name: float(mf(value)) for name, mf in self.mfs.items()}

    def get_universe_array(self) -> np.ndarray:
        """Get discretized universe of discourse."""
        return np.linspace(self.universe[0], self.universe[1], self.resolution)

    def __repr__(self) -> str:
        mf_names = list(self.mfs.keys())
        return f"FuzzyVariable(name='{self.name}', sets={mf_names})"


@dataclass
class FuzzyRule:
    """
    A fuzzy IF-THEN rule for the inference system.

    Structure:
        IF (antecedent_1 AND antecedent_2 AND ...) THEN consequent

    Antecedent Format:
        List of tuples: [(variable_name, fuzzy_set_name), ...]
        Example: [('model1_score', 'positive'), ('model2_score', 'positive')]

    Consequent Format:
        Tuple: (output_variable_name, fuzzy_set_name)
        Example: ('sentiment', 'positive')

    Attributes:
        antecedents: List of (variable, fuzzy_set) pairs
        consequent: Output (variable, fuzzy_set) pair
        weight: Rule importance weight [0, 1]
        operator: How to combine antecedents ('AND' or 'OR')
    """
    antecedents: List[Tuple[str, str]]
    consequent: Tuple[str, str]
    weight: float = 1.0
    operator: str = 'AND'

    def __post_init__(self):
        """Validate rule configuration."""
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Rule weight must be in [0, 1], got {self.weight}")
        if self.operator not in ('AND', 'OR'):
            raise ValueError(f"Operator must be 'AND' or 'OR', got {self.operator}")

    def evaluate(
        self,
        fuzzified_inputs: Dict[str, Dict[str, float]],
        t_norm: FuzzyOperator = FuzzyOperator.MIN,
        t_conorm: FuzzyOperator = FuzzyOperator.MAX
    ) -> float:
        """
        Evaluate the rule given fuzzified input values.

        Parameters
        ----------
        fuzzified_inputs : dict
            Nested dict: {variable_name: {fuzzy_set: membership_degree}}
        t_norm : FuzzyOperator
            T-norm for AND operations
        t_conorm : FuzzyOperator
            T-conorm for OR operations

        Returns
        -------
        float
            Firing strength of this rule (weighted)
        """
        if not self.antecedents:
            return self.weight

        # Get membership degrees for each antecedent
        degrees = []
        for var_name, set_name in self.antecedents:
            if var_name not in fuzzified_inputs:
                raise KeyError(f"Variable '{var_name}' not found in inputs")
            if set_name not in fuzzified_inputs[var_name]:
                raise KeyError(f"Fuzzy set '{set_name}' not found for variable '{var_name}'")
            degrees.append(fuzzified_inputs[var_name][set_name])

        # Combine using appropriate operator
        if self.operator == 'AND':
            result = degrees[0]
            for d in degrees[1:]:
                result = fuzzy_and(result, d, t_norm)
        else:  # OR
            result = degrees[0]
            for d in degrees[1:]:
                result = fuzzy_or(result, d, t_conorm)

        return result * self.weight

    def __repr__(self) -> str:
        ant_str = f" {self.operator} ".join(
            [f"{var} IS {fset}" for var, fset in self.antecedents]
        )
        cons_var, cons_set = self.consequent
        return f"IF {ant_str} THEN {cons_var} IS {cons_set} (w={self.weight})"


class FuzzyInferenceSystem:
    """
    Mamdani-type Fuzzy Inference System for sentiment classification.

    This FIS combines multiple input signals (e.g., from different ML models)
    using fuzzy rules to produce a robust sentiment classification that
    explicitly handles uncertainty.

    Architecture:
        Input Variables → Fuzzification → Rule Evaluation →
        Aggregation → Defuzzification → Output

    Key Features:
        - Multiple input variables (ensemble-ready)
        - Configurable fuzzy operators
        - Rule weight support
        - Multiple aggregation methods

    Example:
        >>> fis = FuzzyInferenceSystem()
        >>> fis.add_input_variable(model1_var)
        >>> fis.add_input_variable(model2_var)
        >>> fis.add_output_variable(output_var)
        >>> fis.add_rule(rule1)
        >>> result = fis.evaluate({'model1': 0.7, 'model2': 0.8})
    """

    def __init__(
        self,
        t_norm: FuzzyOperator = FuzzyOperator.MIN,
        t_conorm: FuzzyOperator = FuzzyOperator.MAX,
        aggregation: str = 'max',
        implication: str = 'min'
    ):
        """
        Initialize the Fuzzy Inference System.

        Parameters
        ----------
        t_norm : FuzzyOperator
            T-norm for AND operations in rules
        t_conorm : FuzzyOperator
            T-conorm for OR operations in rules
        aggregation : str
            Method to aggregate rule outputs ('max', 'sum', 'probor')
        implication : str
            Implication method ('min' for Mamdani, 'product' for Larsen)
        """
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []

        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.aggregation = aggregation
        self.implication = implication

    def add_input_variable(self, variable: FuzzyVariable) -> 'FuzzyInferenceSystem':
        """
        Add an input variable to the system.

        Parameters
        ----------
        variable : FuzzyVariable
            Input linguistic variable

        Returns
        -------
        FuzzyInferenceSystem
            Self for method chaining
        """
        self.input_variables[variable.name] = variable
        return self

    def add_output_variable(self, variable: FuzzyVariable) -> 'FuzzyInferenceSystem':
        """
        Add an output variable to the system.

        Parameters
        ----------
        variable : FuzzyVariable
            Output linguistic variable

        Returns
        -------
        FuzzyInferenceSystem
            Self for method chaining
        """
        self.output_variables[variable.name] = variable
        return self

    def add_rule(self, rule: FuzzyRule) -> 'FuzzyInferenceSystem':
        """
        Add a fuzzy rule to the system.

        Parameters
        ----------
        rule : FuzzyRule
            Fuzzy IF-THEN rule

        Returns
        -------
        FuzzyInferenceSystem
            Self for method chaining
        """
        self.rules.append(rule)
        return self

    def add_rules(self, rules: List[FuzzyRule]) -> 'FuzzyInferenceSystem':
        """Add multiple rules at once."""
        self.rules.extend(rules)
        return self

    def fuzzify_inputs(
        self,
        crisp_inputs: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Fuzzify all input values.

        Parameters
        ----------
        crisp_inputs : dict
            Mapping from variable names to crisp values

        Returns
        -------
        dict
            Nested dict of fuzzified values
        """
        fuzzified = {}
        for var_name, var in self.input_variables.items():
            if var_name not in crisp_inputs:
                raise KeyError(f"Input '{var_name}' required but not provided")
            fuzzified[var_name] = var.fuzzify(crisp_inputs[var_name])
        return fuzzified

    def evaluate_rules(
        self,
        fuzzified_inputs: Dict[str, Dict[str, float]]
    ) -> List[Tuple[FuzzyRule, float]]:
        """
        Evaluate all rules and return their firing strengths.

        Parameters
        ----------
        fuzzified_inputs : dict
            Fuzzified input values

        Returns
        -------
        list
            List of (rule, firing_strength) tuples
        """
        results = []
        for rule in self.rules:
            strength = rule.evaluate(fuzzified_inputs, self.t_norm, self.t_conorm)
            results.append((rule, strength))
        return results

    def aggregate_outputs(
        self,
        rule_evaluations: List[Tuple[FuzzyRule, float]],
        output_var_name: str
    ) -> np.ndarray:
        """
        Aggregate rule outputs into a single fuzzy set.

        Parameters
        ----------
        rule_evaluations : list
            Results from evaluate_rules()
        output_var_name : str
            Name of the output variable

        Returns
        -------
        np.ndarray
            Aggregated membership function values
        """
        output_var = self.output_variables[output_var_name]
        universe = output_var.get_universe_array()
        aggregated = np.zeros_like(universe)

        for rule, firing_strength in rule_evaluations:
            # Only process rules for this output variable
            cons_var, cons_set = rule.consequent
            if cons_var != output_var_name:
                continue

            if firing_strength == 0:
                continue

            # Get the consequent membership function
            cons_mf = output_var.mfs[cons_set]
            cons_values = cons_mf(universe)

            # Apply implication (clip or scale the consequent)
            if self.implication == 'min':
                implied = np.minimum(cons_values, firing_strength)
            else:  # product
                implied = cons_values * firing_strength

            # Aggregate with previous rules
            if self.aggregation == 'max':
                aggregated = np.maximum(aggregated, implied)
            elif self.aggregation == 'sum':
                aggregated = aggregated + implied
            elif self.aggregation == 'probor':
                aggregated = aggregated + implied - aggregated * implied

        return aggregated

    def evaluate(
        self,
        crisp_inputs: Dict[str, float],
        return_details: bool = False
    ) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict]]:
        """
        Perform complete fuzzy inference.

        Parameters
        ----------
        crisp_inputs : dict
            Crisp input values for each input variable
        return_details : bool
            If True, return intermediate results

        Returns
        -------
        dict or tuple
            Aggregated output fuzzy sets (and details if requested)
        """
        # Step 1: Fuzzify inputs
        fuzzified = self.fuzzify_inputs(crisp_inputs)

        # Step 2: Evaluate rules
        rule_evals = self.evaluate_rules(fuzzified)

        # Step 3: Aggregate outputs
        aggregated_outputs = {}
        for output_name in self.output_variables:
            aggregated_outputs[output_name] = self.aggregate_outputs(
                rule_evals, output_name
            )

        if return_details:
            details = {
                'fuzzified_inputs': fuzzified,
                'rule_evaluations': [(str(r), s) for r, s in rule_evals],
            }
            return aggregated_outputs, details

        return aggregated_outputs

    def get_rule_summary(self) -> str:
        """Get a formatted summary of all rules."""
        lines = [f"Fuzzy Inference System with {len(self.rules)} rules:"]
        lines.append(f"  Inputs: {list(self.input_variables.keys())}")
        lines.append(f"  Outputs: {list(self.output_variables.keys())}")
        lines.append(f"  T-norm: {self.t_norm.value}, T-conorm: {self.t_conorm.value}")
        lines.append("\nRules:")
        for i, rule in enumerate(self.rules, 1):
            lines.append(f"  R{i}: {rule}")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (
            f"FuzzyInferenceSystem("
            f"inputs={list(self.input_variables.keys())}, "
            f"outputs={list(self.output_variables.keys())}, "
            f"rules={len(self.rules)})"
        )


# =============================================================================
# Pre-configured Sentiment Inference Systems
# =============================================================================

def create_sentiment_fis(
    input_variables: List[str],
    mf_type: str = 'triangular',
    include_confidence: bool = True
) -> FuzzyInferenceSystem:
    """
    Create a pre-configured FIS for sentiment classification.

    This creates a complete fuzzy inference system with:
    - Input variables for each model's sentiment scores
    - Output variable for final sentiment
    - Standard sentiment classification rules

    Parameters
    ----------
    input_variables : list
        Names of input variables (e.g., ['logreg', 'svm', 'bert'])
    mf_type : str
        Type of membership functions ('triangular', 'gaussian', 'trapezoidal')
    include_confidence : bool
        Whether to add a confidence output variable

    Returns
    -------
    FuzzyInferenceSystem
        Configured inference system ready for use

    Example
    -------
    >>> fis = create_sentiment_fis(['logreg', 'svm'], mf_type='gaussian')
    >>> result = fis.evaluate({'logreg': 0.8, 'svm': 0.75})
    """
    from .membership_functions import create_three_class_mfs

    fis = FuzzyInferenceSystem()

    # Create input variables
    for var_name in input_variables:
        mfs = create_three_class_mfs(mf_type=mf_type)
        input_var = FuzzyVariable(
            name=var_name,
            universe=(0.0, 1.0),
            mfs=mfs
        )
        fis.add_input_variable(input_var)

    # Create output variable for sentiment
    output_mfs = create_three_class_mfs(mf_type=mf_type)
    output_var = FuzzyVariable(
        name='sentiment',
        universe=(0.0, 1.0),
        mfs=output_mfs
    )
    fis.add_output_variable(output_var)

    # Create rules: Generate rules for all combinations
    sentiments = ['negative', 'neutral', 'positive']

    if len(input_variables) == 1:
        # Single input: direct mapping
        var = input_variables[0]
        for sent in sentiments:
            fis.add_rule(FuzzyRule(
                antecedents=[(var, sent)],
                consequent=('sentiment', sent),
                weight=1.0
            ))
    else:
        # Multiple inputs: majority voting style rules
        for sent in sentiments:
            # All agree
            fis.add_rule(FuzzyRule(
                antecedents=[(var, sent) for var in input_variables],
                consequent=('sentiment', sent),
                weight=1.0
            ))

        # Add conflict resolution rules (first two inputs)
        if len(input_variables) >= 2:
            var1, var2 = input_variables[0], input_variables[1]

            # Positive-Neutral → Positive (slight)
            fis.add_rule(FuzzyRule(
                antecedents=[(var1, 'positive'), (var2, 'neutral')],
                consequent=('sentiment', 'positive'),
                weight=0.7
            ))
            fis.add_rule(FuzzyRule(
                antecedents=[(var1, 'neutral'), (var2, 'positive')],
                consequent=('sentiment', 'positive'),
                weight=0.7
            ))

            # Negative-Neutral → Negative (slight)
            fis.add_rule(FuzzyRule(
                antecedents=[(var1, 'negative'), (var2, 'neutral')],
                consequent=('sentiment', 'negative'),
                weight=0.7
            ))
            fis.add_rule(FuzzyRule(
                antecedents=[(var1, 'neutral'), (var2, 'negative')],
                consequent=('sentiment', 'negative'),
                weight=0.7
            ))

            # Positive-Negative conflict → Neutral
            fis.add_rule(FuzzyRule(
                antecedents=[(var1, 'positive'), (var2, 'negative')],
                consequent=('sentiment', 'neutral'),
                weight=0.8
            ))
            fis.add_rule(FuzzyRule(
                antecedents=[(var1, 'negative'), (var2, 'positive')],
                consequent=('sentiment', 'neutral'),
                weight=0.8
            ))

    return fis
