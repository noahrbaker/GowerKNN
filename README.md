# GowerKNN

**High-Performance Gower Distance K-Nearest Neighbors Implementation**

## Overview

GowerKNN is a specialized Python library designed for performing K-Nearest Neighbors (KNN) searches on mixed-type datasets containing both numerical and categorical variables. It implements the Gower Distance metric in a highly optimized, vectorized format suitable for large-scale clinical and epidemiological research.

Standard Euclidean distance metrics fail when applied to nominal data, while standard One-Hot Encoding approaches can suffer from the curse of dimensionality. GowerKNN addresses this by computing similarity directly on mixed data types with support for missing values (NaNs), feature weighting, and massive dataset scaling.

### Key Features

* **Mixed Data Support:** Natively handles continuous, nominal, and boolean data without preprocessing or one-hot encoding.
* **Computational Efficiency:** Utilizes `float32` precision, vectorized Numpy operations, and `argpartition` (Introselect) sorting to achieve high throughput.
* **Memory Optimization:** Implements batched distance computation to process datasets larger than available RAM.
* **Bias Mitigation:** Includes randomization logic during data ingestion to ensuring statistical tie-breaking.
* **Missing Data Handling:** Adheres to Gower's original definition for handling missing data by dynamically adjusting weight normalization, rather than imputing values.

## Installation

GowerKNN utilizes `flit` for packaging. You can install it directly from the source.

"""bash
git clone https://github.com/noahrbaker/GowerKNN
cd GowerKNN
pip install .
"""

## Getting Started

### Basic Usage

The library follows the standard Scikit-Learn estimator API. It automatically detects numerical and categorical columns if a Pandas DataFrame is provided.

```python
import pandas as pd
import numpy as np
from gowerknn import GowerKNN

# 1. Prepare Data (Mixed Types)
data = pd.DataFrame({
    'age': [25, 30, 45, 22, 30],
    'bmi': [22.0, 24.5, 30.1, 19.5, 24.5],
    'sex': ['M', 'F', 'M', 'F', 'F'],
    'smoker': [False, False, True, False, True]
})

# 2. Initialize and Fit
model = GowerKNN()
model.fit(data)

# 3. Query for Neighbors
# Find the single best match for the first patient in the dataset
query = data.iloc[0:1]
distances, indices = model.kneighbors(query, k=1)

print(f"Query Index: {query.index[0]}")
print(f"Matched Index: {indices[0][0]}")
print(f"Gower Distance: {distances[0][0]:.4f}")
```

### Weighted Matching

In clinical matching, certain features (e.g., Sex or Age) may be more critical for control selection than others. You can enforce this using the `weights` parameter.

```python
# Weights align with the columns in the training DataFrame.
# Example: Weighting Age (col 0) and Sex (col 2) as 2x more important.
weights = [2.0, 1.0, 2.0, 1.0]

model = GowerKNN(weights=weights)
model.fit(data)
```

### Advanced Configuration

For large-scale applications (>1 million rows), performance can be tuned using batch sizes and fast sorting algorithms.

```python
# batch_size: Controls memory footprint (rows processed per loop).
# fast_sort: Uses O(N) selection complexity instead of O(N log N).
distances, indices = model.kneighbors(
    query,
    k=10,
    batch_size=1024,
    fast_sort=True
)
```

## Algorithm Details

Gower Distance calculates the dissimilarity between two observations $i$ and $j$ as the weighted average of dissimilarities across all variables $k$.

The distance $D_{ij}$ is defined as:

$$D_{ij} = \frac{\sum_{k} w_k \delta_{ijk} d_{ijk}}{\sum_{k} w_k \delta_{ijk}}$$

Where:
* $w_k$: The weight assigned to variable $k$.
* $\delta_{ijk}$: A binary indicator that is 0 if variable $k$ is missing in either observation, and 1 otherwise.
* $d_{ijk}$: The component distance for variable $k$.
    * **Numerical:** $|x_{ik} - x_{jk}| / R_k$ (where $R_k$ is the range of variable $k$).
    * **Categorical:** 0 if $x_{ik} = x_{jk}$, 1 otherwise.

## API Reference

### `GowerKNN(weights=None, cat_features=None, logger=None)`

* **weights**: *array-like, optional*. Relative importance of each feature.
* **cat_features**: *list or mask, optional*. Explicitly define which columns are categorical. If None, types are inferred from the DataFrame.

### `fit(X, y=None, seed=42)`

Fits the estimator to the reference data `X`.
* **seed**: *int*. Controls the random shuffle of data ingestion to ensuring reproducible tie-breaking.

### `kneighbors(query, k=None, return_distance=True, fast_sort=True, batch_size=512)`

Finds the K-nearest neighbors for the query set.
* **query**: *DataFrame or array*. The subjects to find matches for.
* **k**: *int*. Number of neighbors to retrieve.
* **fast_sort**: *bool*. If True, uses `np.argpartition` for faster query times on large datasets.
* **batch_size**: *int*. Number of query rows to process in a single vectorized block.

## License

This project is distributed under the MIT License. See `LICENSE` for details.

## Citation

If you use this software in academic research, please cite:

> Baker, N. (2026). GowerKNN: High-Performance Gower Distance for Clinical Informatics. University of California, San Francisco.