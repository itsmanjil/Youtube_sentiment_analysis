"""
Defuzzification Methods for Fuzzy Sentiment Classification

This module implements various defuzzification techniques to convert
fuzzy output sets back to crisp values.

Defuzzification is the final step in fuzzy inference, mapping the
aggregated fuzzy output to a single representative value.

Supported Methods:
    1. Centroid (Center of Gravity): Most common, weighted average
    2. Bisector: Point dividing area into two equal parts
    3. MOM (Mean of Maximum): Average of maximum membership points
    4. SOM (Smallest of Maximum): Leftmost maximum
    5. LOM (Largest of Maximum): Rightmost maximum
    6. Weighted Average: For Sugeno-type systems

Theoretical Consideration:
    The choice of defuzzification method affects the system's behavior:
    - Centroid: Smooth, considers entire fuzzy set
    - MOM/SOM/LOM: Focuses on most certain regions
    - Bisector: Balance between extremes

Reference:
    Lee, C.C. (1990). "Fuzzy Logic in Control Systems: Fuzzy Logic
    Controller—Part I". IEEE Transactions on Systems, Man, and Cybernetics.

Author: [Your Name]
"""

from typing import Tuple, Optional, Dict, List
from enum import Enum
import numpy as np

# NumPy 2.0+ compatibility: trapz was renamed to trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


class DefuzzMethod(Enum):
    """Enumeration of defuzzification methods."""
    CENTROID = 'centroid'
    BISECTOR = 'bisector'
    MOM = 'mom'  # Mean of Maximum
    SOM = 'som'  # Smallest of Maximum
    LOM = 'lom'  # Largest of Maximum
    WEIGHTED_AVERAGE = 'weighted_average'


class Defuzzifier:
    """
    Defuzzification engine for converting fuzzy outputs to crisp values.

    This class provides multiple defuzzification methods and can be
    configured to use different approaches based on the application.

    Attributes:
        method: Default defuzzification method
        resolution: Number of points for numerical integration

    Example:
        >>> defuzz = Defuzzifier(method='centroid')
        >>> universe = np.linspace(0, 1, 100)
        >>> membership = compute_aggregated_membership(...)
        >>> crisp_value = defuzz.defuzzify(universe, membership)
    """

    def __init__(
        self,
        method: str = 'centroid',
        resolution: int = 100
    ):
        """
        Initialize the defuzzifier.

        Parameters
        ----------
        method : str
            Default defuzzification method:
            'centroid', 'bisector', 'mom', 'som', 'lom', 'weighted_average'
        resolution : int
            Number of points for numerical calculations
        """
        self.method = DefuzzMethod(method.lower())
        self.resolution = resolution

    def defuzzify(
        self,
        universe: np.ndarray,
        membership: np.ndarray,
        method: Optional[str] = None
    ) -> float:
        """
        Convert a fuzzy set to a crisp value.

        Parameters
        ----------
        universe : np.ndarray
            Array of x values (universe of discourse)
        membership : np.ndarray
            Array of membership degrees corresponding to universe
        method : str, optional
            Override default method for this call

        Returns
        -------
        float
            Crisp defuzzified value

        Raises
        ------
        ValueError
            If arrays have different shapes or if membership is all zeros
        """
        if universe.shape != membership.shape:
            raise ValueError(
                f"Universe and membership must have same shape: "
                f"{universe.shape} vs {membership.shape}"
            )

        # Handle empty or zero membership
        if np.sum(membership) == 0:
            # Return center of universe if no membership
            return (universe[0] + universe[-1]) / 2

        use_method = DefuzzMethod(method.lower()) if method else self.method

        if use_method == DefuzzMethod.CENTROID:
            return self._centroid(universe, membership)
        elif use_method == DefuzzMethod.BISECTOR:
            return self._bisector(universe, membership)
        elif use_method == DefuzzMethod.MOM:
            return self._mean_of_maximum(universe, membership)
        elif use_method == DefuzzMethod.SOM:
            return self._smallest_of_maximum(universe, membership)
        elif use_method == DefuzzMethod.LOM:
            return self._largest_of_maximum(universe, membership)
        elif use_method == DefuzzMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(universe, membership)
        else:
            raise ValueError(f"Unknown defuzzification method: {use_method}")

    def _centroid(self, universe: np.ndarray, membership: np.ndarray) -> float:
        """
        Centroid (Center of Gravity) defuzzification.

        Formula:
            x* = ∫ x·μ(x) dx / ∫ μ(x) dx

        This is the most commonly used method, providing a weighted
        average that considers the entire fuzzy set.

        Properties:
            - Continuous: Small changes in membership → small changes in output
            - Considers entire fuzzy set
            - May produce values not in the maximum membership region
        """
        numerator = _trapz(universe * membership, universe)
        denominator = _trapz(membership, universe)

        if denominator == 0:
            return (universe[0] + universe[-1]) / 2

        return numerator / denominator

    def _bisector(self, universe: np.ndarray, membership: np.ndarray) -> float:
        """
        Bisector defuzzification.

        Finds the point that divides the area under the membership
        function into two equal parts.

        Formula:
            Find x* such that: ∫[a,x*] μ(x) dx = ∫[x*,b] μ(x) dx

        Properties:
            - Considers the shape of the fuzzy set
            - Guaranteed to be within the support of the fuzzy set
        """
        total_area = _trapz(membership, universe)
        half_area = total_area / 2

        # Cumulative area from left
        cumulative = np.zeros_like(membership)
        for i in range(1, len(universe)):
            cumulative[i] = cumulative[i-1] + _trapz(
                membership[i-1:i+1], universe[i-1:i+1]
            )

        # Find where cumulative area crosses half
        idx = np.searchsorted(cumulative, half_area)
        idx = min(idx, len(universe) - 1)

        return universe[idx]

    def _mean_of_maximum(self, universe: np.ndarray, membership: np.ndarray) -> float:
        """
        Mean of Maximum (MOM) defuzzification.

        Returns the average of all points where membership is maximum.

        Formula:
            x* = mean({x : μ(x) = max(μ)})

        Properties:
            - Focuses on the most certain region
            - Ignores the shape of the fuzzy set outside maximum
            - May produce discontinuous outputs
        """
        max_membership = np.max(membership)

        # Allow small tolerance for numerical precision
        tolerance = max_membership * 0.001
        max_indices = np.where(membership >= max_membership - tolerance)[0]

        if len(max_indices) == 0:
            return (universe[0] + universe[-1]) / 2

        return np.mean(universe[max_indices])

    def _smallest_of_maximum(self, universe: np.ndarray, membership: np.ndarray) -> float:
        """
        Smallest of Maximum (SOM) defuzzification.

        Returns the smallest value where membership is maximum.

        Properties:
            - Conservative estimate
            - Useful when smaller values are preferred
        """
        max_membership = np.max(membership)
        tolerance = max_membership * 0.001
        max_indices = np.where(membership >= max_membership - tolerance)[0]

        if len(max_indices) == 0:
            return universe[0]

        return universe[max_indices[0]]

    def _largest_of_maximum(self, universe: np.ndarray, membership: np.ndarray) -> float:
        """
        Largest of Maximum (LOM) defuzzification.

        Returns the largest value where membership is maximum.

        Properties:
            - Optimistic estimate
            - Useful when larger values are preferred
        """
        max_membership = np.max(membership)
        tolerance = max_membership * 0.001
        max_indices = np.where(membership >= max_membership - tolerance)[0]

        if len(max_indices) == 0:
            return universe[-1]

        return universe[max_indices[-1]]

    def _weighted_average(self, universe: np.ndarray, membership: np.ndarray) -> float:
        """
        Weighted Average defuzzification.

        Similar to centroid but uses simple weighted average without
        integration. Typically used with Sugeno-type systems.

        Formula:
            x* = Σ(x_i · μ_i) / Σ(μ_i)
        """
        total_weight = np.sum(membership)
        if total_weight == 0:
            return (universe[0] + universe[-1]) / 2

        return np.sum(universe * membership) / total_weight

    def defuzzify_to_class(
        self,
        universe: np.ndarray,
        membership: np.ndarray,
        class_boundaries: List[Tuple[float, str]],
        method: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Defuzzify and map to a discrete class label.

        Useful for converting continuous fuzzy output to discrete
        sentiment categories (Negative, Neutral, Positive).

        Parameters
        ----------
        universe : np.ndarray
            Array of x values
        membership : np.ndarray
            Array of membership degrees
        class_boundaries : list
            List of (threshold, class_name) tuples, sorted by threshold
            Example: [(0.33, 'negative'), (0.66, 'neutral'), (1.0, 'positive')]
        method : str, optional
            Defuzzification method to use

        Returns
        -------
        tuple
            (class_label, crisp_value)

        Example
        -------
        >>> boundaries = [(0.33, 'negative'), (0.66, 'neutral'), (1.0, 'positive')]
        >>> label, score = defuzz.defuzzify_to_class(u, m, boundaries)
        """
        crisp_value = self.defuzzify(universe, membership, method)

        # Find the class
        for threshold, class_name in class_boundaries:
            if crisp_value <= threshold:
                return class_name, crisp_value

        # Return last class if above all thresholds
        return class_boundaries[-1][1], crisp_value

    def defuzzify_multi_output(
        self,
        outputs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        method: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Defuzzify multiple output variables.

        Parameters
        ----------
        outputs : dict
            Dictionary mapping variable names to (universe, membership) tuples
        method : str, optional
            Defuzzification method to use

        Returns
        -------
        dict
            Dictionary mapping variable names to crisp values
        """
        return {
            name: self.defuzzify(universe, membership, method)
            for name, (universe, membership) in outputs.items()
        }


def compute_uncertainty_metrics(
    universe: np.ndarray,
    membership: np.ndarray
) -> Dict[str, float]:
    """
    Compute uncertainty metrics from a fuzzy output set.

    These metrics help quantify the uncertainty in the fuzzy classification,
    which is valuable for thesis-level analysis.

    Parameters
    ----------
    universe : np.ndarray
        Universe of discourse
    membership : np.ndarray
        Membership degrees

    Returns
    -------
    dict
        Dictionary containing:
        - 'fuzziness': Measure of overall uncertainty [0, 1]
        - 'specificity': Inverse of spread [0, 1]
        - 'entropy': Shannon entropy of membership
        - 'support_width': Width of non-zero membership region
        - 'core_width': Width of full membership region (μ = 1)
        - 'max_membership': Maximum membership degree
    """
    # Normalize membership to sum to 1 for entropy calculation
    total = np.sum(membership)
    if total > 0:
        normalized = membership / total
    else:
        normalized = np.ones_like(membership) / len(membership)

    # Fuzziness index (average distance from crisp set)
    # For crisp sets, membership is 0 or 1, fuzziness = 0
    fuzziness = np.mean(2 * np.minimum(membership, 1 - membership))

    # Specificity (inverse of spread)
    support_indices = np.where(membership > 0.01)[0]
    if len(support_indices) > 0:
        support_width = universe[support_indices[-1]] - universe[support_indices[0]]
        universe_width = universe[-1] - universe[0]
        specificity = 1 - (support_width / universe_width) if universe_width > 0 else 1
    else:
        support_width = 0
        specificity = 1

    # Core width (where membership = 1)
    core_indices = np.where(membership >= 0.99)[0]
    if len(core_indices) > 0:
        core_width = universe[core_indices[-1]] - universe[core_indices[0]]
    else:
        core_width = 0

    # Shannon entropy (using normalized membership as probability)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_norm = np.log2(normalized + 1e-10)
        entropy = -np.sum(normalized * log_norm)

    # Maximum membership degree
    max_membership = np.max(membership)

    return {
        'fuzziness': float(fuzziness),
        'specificity': float(specificity),
        'entropy': float(entropy),
        'support_width': float(support_width),
        'core_width': float(core_width),
        'max_membership': float(max_membership)
    }


def compare_defuzzification_methods(
    universe: np.ndarray,
    membership: np.ndarray
) -> Dict[str, float]:
    """
    Compare all defuzzification methods for analysis.

    Useful for thesis experiments comparing method behaviors.

    Parameters
    ----------
    universe : np.ndarray
        Universe of discourse
    membership : np.ndarray
        Membership degrees

    Returns
    -------
    dict
        Crisp values for each defuzzification method
    """
    defuzz = Defuzzifier()

    return {
        'centroid': defuzz.defuzzify(universe, membership, 'centroid'),
        'bisector': defuzz.defuzzify(universe, membership, 'bisector'),
        'mom': defuzz.defuzzify(universe, membership, 'mom'),
        'som': defuzz.defuzzify(universe, membership, 'som'),
        'lom': defuzz.defuzzify(universe, membership, 'lom'),
        'weighted_average': defuzz.defuzzify(universe, membership, 'weighted_average'),
    }
