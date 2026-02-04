"""
GowerKNN: A high-performance Gower Distance KNN implementation.
GowerKNN provides efficient computation of Gower distances and K-Nearest Neighbors
searching for datasets with mixed data types (numerical and categorical).
"""

__version__ = "0.1.0"

from .matcher import GowerKNN

__all__ = ["GowerKNN"]