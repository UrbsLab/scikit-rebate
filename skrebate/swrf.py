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
from .scoring_utils import SWRF_compute_scores


class SWRF(ReliefF):
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
