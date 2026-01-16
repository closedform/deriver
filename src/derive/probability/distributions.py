"""
distributions.py - Probability Distributions.

Provides probability distribution constructors and functions.
"""

from typing import Any, Optional, Callable, Union
import numpy as np
from scipy import stats as scipy_stats


def NormalDistribution(mu: float = 0, sigma: float = 1):
    """
    Normal distribution.

    Returns a scipy.stats distribution object.

    Args:
        mu: Mean (default 0)
        sigma: Standard deviation (default 1)

    Returns:
        scipy.stats normal distribution object

    Examples:
        >>> dist = NormalDistribution(0, 1)
        >>> PDF(dist, 0)  # Value at x=0
        >>> CDF(dist, 0)  # Cumulative at x=0
    """
    return scipy_stats.norm(loc=mu, scale=sigma)


def UniformDistribution(a: float = 0, b: float = 1):
    """
    Uniform distribution on [a, b].

    Args:
        a: Lower bound (default 0)
        b: Upper bound (default 1)

    Returns:
        scipy.stats uniform distribution object
    """
    return scipy_stats.uniform(loc=a, scale=b-a)


def ExponentialDistribution(lambd: float = 1):
    """
    Exponential distribution with rate lambda.

    Args:
        lambd: Rate parameter (default 1)

    Returns:
        scipy.stats exponential distribution object
    """
    return scipy_stats.expon(scale=1/lambd)


def PoissonDistribution(mu: float):
    """
    Poisson distribution with mean mu.

    Args:
        mu: Mean (rate parameter)

    Returns:
        scipy.stats Poisson distribution object
    """
    return scipy_stats.poisson(mu=mu)


def BinomialDistribution(n: int, p: float):
    """
    Binomial distribution with n trials and success probability p.

    Args:
        n: Number of trials
        p: Success probability

    Returns:
        scipy.stats binomial distribution object
    """
    return scipy_stats.binom(n=n, p=p)


def PDF(dist: Any, x: Any) -> float:
    """
    Probability Density Function (or PMF for discrete).

    PDF[dist, x] - evaluate PDF/PMF at x

    Args:
        dist: Distribution object
        x: Point to evaluate at

    Returns:
        PDF/PMF value at x

    Examples:
        >>> dist = NormalDistribution(0, 1)
        >>> PDF(dist, 0)
        0.3989422804014327
    """
    if hasattr(dist, 'pdf'):
        return dist.pdf(x)
    elif hasattr(dist, 'pmf'):
        return dist.pmf(x)
    raise ValueError("Distribution must have pdf or pmf method")


def CDF(dist: Any, x: Any) -> float:
    """
    Cumulative Distribution Function.

    CDF[dist, x] - P(X <= x)

    Args:
        dist: Distribution object
        x: Point to evaluate at

    Returns:
        CDF value at x

    Examples:
        >>> dist = NormalDistribution(0, 1)
        >>> CDF(dist, 0)
        0.5
    """
    return dist.cdf(x)


def Mean(dist_or_data: Any) -> float:
    """
    Compute mean of distribution or data.

    Args:
        dist_or_data: Distribution object or array-like data

    Returns:
        Mean value

    Examples:
        >>> Mean(NormalDistribution(5, 2))
        5.0
        >>> Mean([1, 2, 3, 4, 5])
        3.0
    """
    if hasattr(dist_or_data, 'mean'):
        return dist_or_data.mean()
    return np.mean(dist_or_data)


def Variance(dist_or_data: Any) -> float:
    """
    Compute variance of distribution or data.

    Args:
        dist_or_data: Distribution object or array-like data

    Returns:
        Variance

    Examples:
        >>> Variance(NormalDistribution(0, 2))
        4.0
    """
    if hasattr(dist_or_data, 'var'):
        return dist_or_data.var()
    return np.var(dist_or_data)


def StandardDeviation(dist_or_data: Any) -> float:
    """
    Compute standard deviation of distribution or data.

    Args:
        dist_or_data: Distribution object or array-like data

    Returns:
        Standard deviation
    """
    if hasattr(dist_or_data, 'std'):
        return dist_or_data.std()
    return np.std(dist_or_data)


def RandomVariate(dist: Any, n: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Generate random variates from distribution.

    RandomVariate[dist] - single random value
    RandomVariate[dist, n] - n random values

    Args:
        dist: Distribution object
        n: Number of samples (default: 1, returns scalar)

    Returns:
        Random sample(s)

    Examples:
        >>> dist = NormalDistribution(0, 1)
        >>> RandomVariate(dist, 5)
        array([...])  # 5 random normal values
    """
    if n is None:
        return dist.rvs()
    return dist.rvs(size=n)


def Probability(condition: Callable, dist: Any) -> float:
    """
    Compute probability of condition for distribution.

    For simple conditions like P(X < a) or P(a < X < b).

    Args:
        condition: Callable that takes x and returns boolean
        dist: Distribution object

    Returns:
        Estimated probability (via Monte Carlo)

    Examples:
        >>> dist = NormalDistribution(0, 1)
        >>> Probability(lambda x: x < 0, dist)  # Approximation
    """
    if callable(condition):
        samples = dist.rvs(size=100000)
        return np.mean(condition(samples))
    raise ValueError("Condition must be callable")


__all__ = [
    'NormalDistribution',
    'UniformDistribution',
    'ExponentialDistribution',
    'PoissonDistribution',
    'BinomialDistribution',
    'PDF',
    'CDF',
    'Mean',
    'Variance',
    'StandardDeviation',
    'RandomVariate',
    'Probability',
]
