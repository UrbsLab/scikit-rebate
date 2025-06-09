# -*- coding: utf-8 -*-

"""
scikit-rebate was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Pete Schmitt (pschmitt@upenn.edu)
    - Ryan J. Urbanowicz (ryanurb@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - and many more generous open source contributors
Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
import numpy as np
from joblib import Parallel, delayed
from .relieff import ReliefF


def SWRF_compute_scores(instance_index, X, y, distance_array, mean_dist, std_dist,
                        nan_mask, class_type, labels_std, data_type,
                        sample_weights=None, mcmap=None):
    """
    Compute the SWRF* contribution to feature scores from a single instance (instance_index).
    Parameters
    ----------
    instance_index : int
        Index of the instance for which to compute score contributions.
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : numpy.ndarray, shape (n_samples,)
        Label vector.
    distance_array : list of numpy.ndarray
        Precomputed lower-triangular distance matrix. distance_array[j][i] gives distance between 
        samples j and i (for j > i). For j < i, distance is in distance_array[i][j].
    mean_dist : float
        Mean of all pairwise distances.
    std_dist : float
        Standard deviation of all pairwise distances.
    nan_mask : numpy.ndarray, shape (n_samples, n_features)
        Boolean mask where True indicates a missing value (np.nan) in X.
    class_type : str
        Type of target variable ('classification' or 'continuous').
    labels_std : float or None
        For continuous targets, a normalization factor (e.g., range or std of y). Unused for classification.
    data_type : any
        Feature type information (not explicitly used in this implementation; assumes diff is Hamming).
    sample_weights : numpy.ndarray or None
        Optional array of sample weights of shape (n_samples,). If provided, the instance's contributions 
        will be scaled by sample_weights[instance_index].
    mcmap : dict or None
        For multi-class classification, a mapping from class labels to their frequencies (or prior probabilities).
    Returns
    -------
    numpy.ndarray, shape (n_features,)
        The contribution of this instance to the global feature scores (vector of feature weight updates).
    """
    n_samples, n_features = X.shape
    i = instance_index

    # Retrieve distances from instance i to all other instances
    dist_i = np.zeros(n_samples, dtype=float)
    for j in range(n_samples):
        if j == i:
            continue
        elif j < i:
            dist_i[j] = distance_array[i][j]  # i > j, stored in row i
        else:
            dist_i[j] = distance_array[j][i]  # j > i, stored in row j

    # Compute sigmoid weights for all neighbors of i
    if std_dist is None or std_dist == 0 or np.isclose(std_dist, 0.0):
        # If no variation in distances (std_dist=0), assign zero weight to all neighbors
        weight_vec = np.zeros(n_samples, dtype=float)
    else:
        # Sigmoid weighting: f(dist) = 2 / (1 + exp((dist - t) * 4 / u)) - 1
        diff = (dist_i - mean_dist) * (4.0 / std_dist)
        # Clamp the exponent to avoid overflow in exp()
        diff = np.clip(diff, -50, 50)
        weight_vec = 2.0 / (1.0 + np.exp(diff)) - 1.0
    # Exclude the instance itself from neighbor contributions
    weight_vec[i] = 0.0

    # Determine class comparison signs (or factors) for each neighbor
    if class_type == 'multiclass':  # classification (binary or multi-class)
        if mcmap is not None and len(mcmap) > 2:
            # Multi-class: weight different-class neighbors by class prior P(c) / (1 - P(class_i))
            total = sum(mcmap.values())
            P_ci = mcmap.get(y[i], 0) / float(total) if total > 0 else 0.0
            class_sign = np.zeros(n_samples, dtype=float)
            # Same-class neighbors: negative sign
            class_sign[y == y[i]] = -1.0
            # Different-class neighbors: assign factor = P(c) / (1 - P(ci)) for class c
            for cls, count in mcmap.items():
                if cls == y[i] or total == 0:
                    continue
                P_c = count / float(total)
                factor = 0.0 if (1 - P_ci) <= 0 else (P_c / (1 - P_ci))
                class_sign[y == cls] = factor
            class_sign[i] = 0.0  # self excluded
    elif class_type == 'binary':
        # Binary classification (or no mcmap): hits = -1, misses = +1
        class_sign = np.where(y == y[i], -1.0, 1.0).astype(float)
        class_sign[i] = 0.0
    else:
        # Continuous target: use normalized difference in outcome as weight (RReliefF style)
        if labels_std is None or labels_std == 0:
            # If no normalization factor, default to no class difference scaling
            class_sign = np.zeros(n_samples, dtype=float)
        else:
            # For regression, we use the difference in response as the "class" factor (absolute difference normalized)
            class_sign = np.abs(y - y[i]) / float(labels_std)
            class_sign[i] = 0.0

    # Combine class sign and distance weight for each neighbor
    neighbor_effect = class_sign * weight_vec  # element-wise multiplication

    # Compute feature-wise differences between instance i and all neighbors
    # diff_matrix[j, a] = 1 if X[i,a] != X[j,a], else 0. Handle missing values as no difference.
    diff_matrix = (X != X[i])  # boolean matrix of shape (n_samples, n_features)
    if nan_mask is not None:
        # Invalidate differences where either value is missing
        valid_mask = (~nan_mask) & (~nan_mask[i])  # True where neither X[i,a] nor X[j,a] is NaN
        diff_matrix = diff_matrix & valid_mask

    # Calculate contributions for each feature: weighted sum of differences over all neighbors
    # This is effectively: sum_j [ neighbor_effect[j] * diff_matrix[j,a] ] for each feature a
    diff_matrix = diff_matrix.astype(float)  # convert bool to float (0.0/1.0)
    feature_updates = neighbor_effect.dot(diff_matrix)  # shape (n_features,)

    # Normalize the update by total weight magnitude to keep it in [-1, 1]
    total_weight = np.sum(np.abs(weight_vec))
    if total_weight > 0:
        feature_updates /= total_weight

    # Scale by sample weight if provided
    if sample_weights is not None:
        feature_updates *= float(sample_weights[i])

    return feature_updates


class SWRFstar(ReliefF):
    """Feature selection using the Sigmoid Weighted ReliefF* algorithm (SWRF*).
    This class implements the SWRF* algorithm, a spatially-weighted version of ReliefF:contentReference[oaicite:15]{index=15}. 
    SWRF* uses a sigmoid neighbor weighting function (centered at the mean distance `t` and width `u` = std. dev.) 
    to weight the contribution of each neighbor. All instances (except self) are considered as neighbors: those closer 
    than `t` have positive weight, and those farther than `t` have negative weight:contentReference[oaicite:16]{index=16}. This allows distant 
    points to inversely influence feature scores, improving detection of complex interactions. The original input/output 
    interface of SURF is maintained, but the underlying scoring is per the SWRF* algorithm.
    """

    def __init__(self, n_features_to_select=10, categorical_features=None, 
                 discrete_threshold=10, verbose=False, n_jobs=1, 
                 weight_final_scores=False, rank_absolute=False, label_type=None):
        """
        Parameters
        ----------
        n_features_to_select : int (default=10)
            Number of top-ranked features to select after Relief-based scoring.
        categorical_features : list of int (default=None)
            Indices of features to treat as categorical (nominal). Features not listed are treated as continuous by default.
        discrete_threshold : int (default=10)
            (Deprecated in SWRF*) Threshold for unique values to consider a feature discrete vs continuous. 
            SWRF* does not use a fixed radius, so this is not actively used.
        verbose : bool (default=False)
            If True, print timing information for distance calculation and scoring.
        n_jobs : int (default=1)
            Number of CPU cores to use for parallel computation. -1 uses all available cores.
        weight_final_scores : bool (default=False)
            If True, multiply final feature scores by the sample weights provided to `fit` (if any).
        rank_absolute : bool (default=False)
            If True, rank features by absolute importance (magnitude) instead of raw scores.
        label_type : str or None (default=None)
            Type of the target variable ('classification' or 'continuous'). If None, will be inferred from y in fit.
        """
        self.n_features_to_select = n_features_to_select
        self.categorical_features = categorical_features
        self.discrete_threshold = discrete_threshold  # not used in SWRF*
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.weight_final_scores = weight_final_scores
        self.rank_absolute = rank_absolute
        self.label_type = label_type

    def _run_algorithm(self):
        """Compute SWRF* feature scores for the dataset."""
        n_samples = self._datalen
        n_features = self._num_attributes

        if n_samples < 2:
            print("Not enough samples to compute feature scores; returning zero scores.")
            # Not enough samples to compare; return zero scores
            return np.zeros(n_features)

        # Calculate mean and std of all pairwise distances in the dataset
        total_sum = 0.0
        total_sum_sq = 0.0
        count = 0
        for i in range(n_samples):
            if len(self._distance_array[i]) > 0:
                # Use vectorized ops for speed
                dists = np.array(self._distance_array[i], dtype=float)
                total_sum += dists.sum()
                total_sum_sq += (dists ** 2).sum()
                count += dists.size
        mean_dist = total_sum / count
        # Compute standard deviation in a numerically stable way
        variance = (total_sum_sq / count) - (mean_dist ** 2)
        std_dist = np.sqrt(variance) if variance > 0 else 0.0

        # Precompute missing values mask
        nan_mask = np.isnan(self._X)

        # Compute feature contributions from each instance in parallel
        if isinstance(self._weights, np.ndarray) and self.weight_final_scores:
            # Include sample weights in the computation
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(SWRF_compute_scores)(
                    i, self._X, self._y, self._distance_array, mean_dist, std_dist, 
                    nan_mask, self._class_type, self._labels_std, self.data_type, 
                    sample_weights=self._weights, mcmap=self.mcmap
                ) for i in range(n_samples)
            )
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(SWRF_compute_scores)(
                    i, self._X, self._y, self._distance_array, mean_dist, std_dist, 
                    nan_mask, self._class_type, self._labels_std, self.data_type, 
                    sample_weights=None, mcmap=self.mcmap
                ) for i in range(n_samples)
            )
        # Sum contributions from all instances to get total feature scores
        feature_scores = np.sum(results, axis=0)
        return np.array(feature_scores)
