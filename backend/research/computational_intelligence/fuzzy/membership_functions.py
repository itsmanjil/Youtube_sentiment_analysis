"""
Membership Functions for Fuzzy Sentiment Classification

This module implements various membership functions used in fuzzy set theory
to map crisp sentiment scores to fuzzy membership degrees.

Mathematical Foundations:
- A membership function μ_A(x) maps elements x to [0, 1]
- μ_A(x) = 1 indicates full membership in fuzzy set A
- μ_A(x) = 0 indicates no membership in fuzzy set A
- 0 < μ_A(x) < 1 indicates partial membership (uncertainty)

Supported Membership Functions:
1. Triangular: Simple, computationally efficient
2. Trapezoidal: Allows plateau region for "definitely in set"
3. Gaussian: Smooth, differentiable (good for gradient-based optimization)
4. Sigmoid: S-shaped, asymmetric boundaries
5. Generalized Bell: Smooth with adjustable shoulders

Reference:
    Zadeh, L.A. (1965). "Fuzzy Sets". Information and Control, 8(3), 338-353.

Author: [Your Name]
"""

from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional
import numpy as np


class MembershipFunction(ABC):
    """
    Abstract base class for all membership functions.

    A membership function defines the degree to which a crisp value
    belongs to a fuzzy set, returning values in the range [0, 1].

    Attributes:
        name (str): Identifier for this membership function
        universe (tuple): The valid input range (min, max)
    """

    def __init__(self, name: str, universe: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize the membership function.

        Parameters
        ----------
        name : str
            A descriptive name for this membership function
        universe : tuple
            The valid input range as (min_value, max_value)
        """
        self.name = name
        self.universe = universe

    @abstractmethod
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the membership degree for input value(s).

        Parameters
        ----------
        x : float or np.ndarray
            Input value(s) to compute membership for

        Returns
        -------
        float or np.ndarray
            Membership degree(s) in range [0, 1]
        """
        pass

    def plot_data(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for plotting the membership function.

        Parameters
        ----------
        num_points : int
            Number of points to generate

        Returns
        -------
        tuple
            (x_values, membership_degrees) for plotting
        """
        x = np.linspace(self.universe[0], self.universe[1], num_points)
        y = self(x)
        return x, y

    def alpha_cut(self, alpha: float, x: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Apply alpha-cut to determine if x belongs to the alpha-level set.

        The alpha-cut A_α contains all elements with membership >= α.

        Parameters
        ----------
        alpha : float
            Threshold value in [0, 1]
        x : float or np.ndarray
            Input value(s)

        Returns
        -------
        bool or np.ndarray
            True if membership(x) >= alpha
        """
        return self(x) >= alpha

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', universe={self.universe})"


class TriangularMF(MembershipFunction):
    """
    Triangular Membership Function.

    Defined by three parameters (a, b, c) where:
    - a: left foot (membership = 0)
    - b: peak (membership = 1)
    - c: right foot (membership = 0)

    Mathematical Definition:
        μ(x) = 0,           if x <= a
        μ(x) = (x-a)/(b-a), if a < x <= b
        μ(x) = (c-x)/(c-b), if b < x < c
        μ(x) = 0,           if x >= c

    Use Case in Sentiment Analysis:
        Ideal for defining "Neutral" sentiment where there's a clear
        peak around 0.5 sentiment score.
    """

    def __init__(
        self,
        name: str,
        a: float,
        b: float,
        c: float,
        universe: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Initialize triangular membership function.

        Parameters
        ----------
        name : str
            Name of the fuzzy set (e.g., "Neutral")
        a : float
            Left foot of triangle
        b : float
            Peak of triangle
        c : float
            Right foot of triangle
        universe : tuple
            Valid input range

        Raises
        ------
        ValueError
            If parameters don't satisfy a <= b <= c
        """
        super().__init__(name, universe)

        if not (a <= b <= c):
            raise ValueError(f"Parameters must satisfy a <= b <= c, got a={a}, b={b}, c={c}")

        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute triangular membership degree."""
        x = np.asarray(x)

        # Initialize output
        result = np.zeros_like(x, dtype=float)

        # Rising edge: a < x <= b
        mask_rise = (x > self.a) & (x <= self.b)
        if self.b != self.a:  # Avoid division by zero
            result[mask_rise] = (x[mask_rise] - self.a) / (self.b - self.a)

        # Falling edge: b < x < c
        mask_fall = (x > self.b) & (x < self.c)
        if self.c != self.b:  # Avoid division by zero
            result[mask_fall] = (self.c - x[mask_fall]) / (self.c - self.b)

        # Handle scalar input
        if result.ndim == 0:
            return float(result)

        return result

    @property
    def parameters(self) -> Tuple[float, float, float]:
        """Return the (a, b, c) parameters."""
        return (self.a, self.b, self.c)


class TrapezoidalMF(MembershipFunction):
    """
    Trapezoidal Membership Function.

    Defined by four parameters (a, b, c, d) where:
    - a: left foot (membership = 0)
    - b: left shoulder (membership = 1 starts)
    - c: right shoulder (membership = 1 ends)
    - d: right foot (membership = 0)

    Mathematical Definition:
        μ(x) = 0,           if x <= a
        μ(x) = (x-a)/(b-a), if a < x < b
        μ(x) = 1,           if b <= x <= c
        μ(x) = (d-x)/(d-c), if c < x < d
        μ(x) = 0,           if x >= d

    Use Case in Sentiment Analysis:
        Ideal for "Strongly Positive" or "Strongly Negative" where
        there's a range of values that are definitely in the set.
    """

    def __init__(
        self,
        name: str,
        a: float,
        b: float,
        c: float,
        d: float,
        universe: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Initialize trapezoidal membership function.

        Parameters
        ----------
        name : str
            Name of the fuzzy set
        a : float
            Left foot
        b : float
            Left shoulder
        c : float
            Right shoulder
        d : float
            Right foot
        universe : tuple
            Valid input range

        Raises
        ------
        ValueError
            If parameters don't satisfy a <= b <= c <= d
        """
        super().__init__(name, universe)

        if not (a <= b <= c <= d):
            raise ValueError(
                f"Parameters must satisfy a <= b <= c <= d, "
                f"got a={a}, b={b}, c={c}, d={d}"
            )

        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute trapezoidal membership degree."""
        x = np.asarray(x)

        result = np.zeros_like(x, dtype=float)

        # Rising edge: a < x < b
        mask_rise = (x > self.a) & (x < self.b)
        if self.b != self.a:
            result[mask_rise] = (x[mask_rise] - self.a) / (self.b - self.a)

        # Plateau: b <= x <= c
        mask_plateau = (x >= self.b) & (x <= self.c)
        result[mask_plateau] = 1.0

        # Falling edge: c < x < d
        mask_fall = (x > self.c) & (x < self.d)
        if self.d != self.c:
            result[mask_fall] = (self.d - x[mask_fall]) / (self.d - self.c)

        if result.ndim == 0:
            return float(result)

        return result

    @property
    def parameters(self) -> Tuple[float, float, float, float]:
        """Return the (a, b, c, d) parameters."""
        return (self.a, self.b, self.c, self.d)


class GaussianMF(MembershipFunction):
    """
    Gaussian Membership Function.

    Defined by mean (μ) and standard deviation (σ):
        μ(x) = exp(-0.5 * ((x - mean) / sigma)^2)

    Properties:
    - Smooth and differentiable everywhere
    - Symmetric around the mean
    - Never reaches exactly 0 (asymptotic)
    - Peak membership of 1 at x = mean

    Use Case in Sentiment Analysis:
        Ideal for modeling uncertainty around sentiment scores,
        especially useful when integrating with neural networks
        (gradient-based optimization).
    """

    def __init__(
        self,
        name: str,
        mean: float,
        sigma: float,
        universe: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Initialize Gaussian membership function.

        Parameters
        ----------
        name : str
            Name of the fuzzy set
        mean : float
            Center of the Gaussian (peak membership)
        sigma : float
            Standard deviation (controls width)
        universe : tuple
            Valid input range

        Raises
        ------
        ValueError
            If sigma <= 0
        """
        super().__init__(name, universe)

        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")

        self.mean = mean
        self.sigma = sigma

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute Gaussian membership degree."""
        x = np.asarray(x)
        result = np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)

        if result.ndim == 0:
            return float(result)

        return result

    @property
    def parameters(self) -> Tuple[float, float]:
        """Return the (mean, sigma) parameters."""
        return (self.mean, self.sigma)


class SigmoidMF(MembershipFunction):
    """
    Sigmoid (S-shaped) Membership Function.

    Mathematical Definition:
        μ(x) = 1 / (1 + exp(-a * (x - c)))

    where:
    - a: controls steepness (positive = rising, negative = falling)
    - c: inflection point (where membership = 0.5)

    Use Case in Sentiment Analysis:
        Ideal for modeling "Positive" sentiment (rising sigmoid) or
        "Negative" sentiment (falling sigmoid) with smooth transitions.
    """

    def __init__(
        self,
        name: str,
        a: float,
        c: float,
        universe: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Initialize sigmoid membership function.

        Parameters
        ----------
        name : str
            Name of the fuzzy set
        a : float
            Steepness parameter (positive = rising S, negative = falling S)
        c : float
            Inflection point (x value where membership = 0.5)
        universe : tuple
            Valid input range
        """
        super().__init__(name, universe)
        self.a = a
        self.c = c

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute sigmoid membership degree."""
        x = np.asarray(x)

        # Use numerically stable sigmoid
        z = self.a * (x - self.c)
        result = np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

        if result.ndim == 0:
            return float(result)

        return result

    @property
    def parameters(self) -> Tuple[float, float]:
        """Return the (a, c) parameters."""
        return (self.a, self.c)


class BellMF(MembershipFunction):
    """
    Generalized Bell Membership Function.

    Mathematical Definition:
        μ(x) = 1 / (1 + |((x - c) / a)|^(2b))

    where:
    - a: width parameter (controls spread)
    - b: slope parameter (controls steepness of sides)
    - c: center (peak membership)

    Properties:
    - Smooth and symmetric
    - Adjustable shoulders (controlled by b)
    - Generalizes both Gaussian and rectangular shapes

    Use Case in Sentiment Analysis:
        Provides more control over the shape than Gaussian,
        useful for fine-tuning fuzzy set boundaries.
    """

    def __init__(
        self,
        name: str,
        a: float,
        b: float,
        c: float,
        universe: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Initialize generalized bell membership function.

        Parameters
        ----------
        name : str
            Name of the fuzzy set
        a : float
            Width parameter (must be non-zero)
        b : float
            Slope parameter (must be positive)
        c : float
            Center of the bell
        universe : tuple
            Valid input range

        Raises
        ------
        ValueError
            If a == 0 or b <= 0
        """
        super().__init__(name, universe)

        if a == 0:
            raise ValueError("Parameter 'a' cannot be zero")
        if b <= 0:
            raise ValueError(f"Parameter 'b' must be positive, got {b}")

        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute generalized bell membership degree."""
        x = np.asarray(x)

        result = 1 / (1 + np.abs((x - self.c) / self.a) ** (2 * self.b))

        if result.ndim == 0:
            return float(result)

        return result

    @property
    def parameters(self) -> Tuple[float, float, float]:
        """Return the (a, b, c) parameters."""
        return (self.a, self.b, self.c)


# =============================================================================
# Utility Functions for Creating Common Sentiment MF Configurations
# =============================================================================

def create_sentiment_mfs_triangular(
    universe: Tuple[float, float] = (0.0, 1.0)
) -> dict:
    """
    Create a standard set of triangular MFs for sentiment analysis.

    Creates five fuzzy sets:
    - Strongly Negative
    - Negative
    - Neutral
    - Positive
    - Strongly Positive

    Parameters
    ----------
    universe : tuple
        The input range for sentiment scores

    Returns
    -------
    dict
        Dictionary mapping sentiment names to MembershipFunction objects
    """
    return {
        'strongly_negative': TriangularMF('Strongly Negative', 0.0, 0.0, 0.2, universe),
        'negative': TriangularMF('Negative', 0.0, 0.25, 0.4, universe),
        'neutral': TriangularMF('Neutral', 0.3, 0.5, 0.7, universe),
        'positive': TriangularMF('Positive', 0.6, 0.75, 1.0, universe),
        'strongly_positive': TriangularMF('Strongly Positive', 0.8, 1.0, 1.0, universe),
    }


def create_sentiment_mfs_gaussian(
    universe: Tuple[float, float] = (0.0, 1.0)
) -> dict:
    """
    Create a standard set of Gaussian MFs for sentiment analysis.

    Uses Gaussian functions for smoother boundaries,
    better suited for gradient-based optimization.

    Parameters
    ----------
    universe : tuple
        The input range for sentiment scores

    Returns
    -------
    dict
        Dictionary mapping sentiment names to MembershipFunction objects
    """
    return {
        'strongly_negative': GaussianMF('Strongly Negative', 0.0, 0.1, universe),
        'negative': GaussianMF('Negative', 0.25, 0.12, universe),
        'neutral': GaussianMF('Neutral', 0.5, 0.12, universe),
        'positive': GaussianMF('Positive', 0.75, 0.12, universe),
        'strongly_positive': GaussianMF('Strongly Positive', 1.0, 0.1, universe),
    }


def create_sentiment_mfs_trapezoidal(
    universe: Tuple[float, float] = (0.0, 1.0)
) -> dict:
    """
    Create a standard set of trapezoidal MFs for sentiment analysis.

    Uses trapezoidal functions for crisp core regions
    with gradual transitions at boundaries.

    Parameters
    ----------
    universe : tuple
        The input range for sentiment scores

    Returns
    -------
    dict
        Dictionary mapping sentiment names to MembershipFunction objects
    """
    return {
        'strongly_negative': TrapezoidalMF('Strongly Negative', 0.0, 0.0, 0.1, 0.2, universe),
        'negative': TrapezoidalMF('Negative', 0.1, 0.2, 0.3, 0.4, universe),
        'neutral': TrapezoidalMF('Neutral', 0.35, 0.45, 0.55, 0.65, universe),
        'positive': TrapezoidalMF('Positive', 0.6, 0.7, 0.8, 0.9, universe),
        'strongly_positive': TrapezoidalMF('Strongly Positive', 0.8, 0.9, 1.0, 1.0, universe),
    }


def create_three_class_mfs(
    mf_type: str = 'triangular',
    universe: Tuple[float, float] = (0.0, 1.0)
) -> dict:
    """
    Create a three-class fuzzy partition (Negative, Neutral, Positive).

    This is the most common configuration for sentiment analysis,
    matching the typical three-class output of ML models.

    Parameters
    ----------
    mf_type : str
        Type of membership function: 'triangular', 'gaussian', or 'trapezoidal'
    universe : tuple
        The input range for sentiment scores

    Returns
    -------
    dict
        Dictionary with 'negative', 'neutral', 'positive' keys
    """
    if mf_type == 'triangular':
        return {
            'negative': TriangularMF('Negative', 0.0, 0.0, 0.5, universe),
            'neutral': TriangularMF('Neutral', 0.25, 0.5, 0.75, universe),
            'positive': TriangularMF('Positive', 0.5, 1.0, 1.0, universe),
        }
    elif mf_type == 'gaussian':
        return {
            'negative': GaussianMF('Negative', 0.0, 0.2, universe),
            'neutral': GaussianMF('Neutral', 0.5, 0.15, universe),
            'positive': GaussianMF('Positive', 1.0, 0.2, universe),
        }
    elif mf_type == 'trapezoidal':
        return {
            'negative': TrapezoidalMF('Negative', 0.0, 0.0, 0.2, 0.4, universe),
            'neutral': TrapezoidalMF('Neutral', 0.3, 0.4, 0.6, 0.7, universe),
            'positive': TrapezoidalMF('Positive', 0.6, 0.8, 1.0, 1.0, universe),
        }
    else:
        raise ValueError(f"Unknown mf_type: {mf_type}. Use 'triangular', 'gaussian', or 'trapezoidal'")
