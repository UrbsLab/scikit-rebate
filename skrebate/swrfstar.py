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


def sigmoid_weight(distances, mean_dist, std_dist):
    diff = (distances - mean_dist) * (4.0 / std_dist)
    diff = np.clip(diff, -50, 50)
    return 2.0 / (1.0 + np.exp(diff)) - 1.0


def ramp_function(data_type, attr, fname, xinstfeature, xNNifeature):
    diff = 0
    mmdiff = attr[fname][3]
    rawfd = abs(xinstfeature - xNNifeature)
    if data_type == 'mixed':
        standDev = attr[fname][4]
        if rawfd > standDev:
            diff = 1
        else:
            diff = rawfd / mmdiff
    else:
        diff = rawfd / mmdiff
    return diff


class SWRFstar(ReliefF):
    def __init__(self, n_features_to_select=10, categorical_features=None,
                 discrete_threshold=10, multiclass_threshold=10, verbose=False, n_jobs=1,
                 weight_final_scores=False, rank_absolute=False, label_type=None):
        self.n_features_to_select = n_features_to_select
        self.categorical_features = categorical_features
        self.discrete_threshold = discrete_threshold
        self.multiclass_threshold = multiclass_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.weight_final_scores = weight_final_scores
        self.rank_absolute = rank_absolute
        self.label_type = label_type

    def _score_instance(self, inst_idx, mean_dist, std_dist, nan_mask):
        n_features = self._num_attributes
        dist_i = np.zeros(self._datalen, dtype=float)
        for j in range(self._datalen):
            if j == inst_idx:
                continue
            elif j < inst_idx:
                dist_i[j] = self._distance_array[inst_idx][j]
            else:
                dist_i[j] = self._distance_array[j][inst_idx]

        sigmoid_weights = sigmoid_weight(dist_i, mean_dist, std_dist)
        sigmoid_weights[inst_idx] = 0.0

        feature_scores = np.zeros(n_features, dtype=float)
        for j in range(self._datalen):
            if j == inst_idx or sigmoid_weights[j] == 0:
                continue
            for a in range(n_features):
                if nan_mask[inst_idx][a] or nan_mask[j][a]:
                    continue
                ftype = self.attr[self._headers[a]][0]
                if ftype == 'continuous':
                    diff = ramp_function(self.data_type, self.attr, self._headers[a],
                                         self._X[inst_idx][a], self._X[j][a])
                else:
                    diff = float(self._X[inst_idx][a] != self._X[j][a])

                if self._class_type == 'binary':
                    # Same class: subtract, Different class: add
                    if self._y[inst_idx] == self._y[j]:
                        feature_scores[a] -= sigmoid_weights[j] * diff
                    else:
                        feature_scores[a] += sigmoid_weights[j] * diff

                elif self._class_type == 'multiclass':
                    if self._y[inst_idx] == self._y[j]:
                        feature_scores[a] -= sigmoid_weights[j] * diff
                    else:
                        P_ci = self.mcmap.get(self._y[inst_idx], 0) / float(sum(self.mcmap.values()))
                        P_cj = self.mcmap.get(self._y[j], 0) / float(sum(self.mcmap.values()))
                        factor = P_cj / (1 - P_ci) if (1 - P_ci) > 0 else 0
                        feature_scores[a] += sigmoid_weights[j] * diff * factor

                else:
                    if self._labels_std is not None and self._labels_std > 0:
                        response_diff = self._y[inst_idx] - self._y[j]
                        similarity = np.exp(-(response_diff ** 2) / (2 * (self._labels_std ** 2)))
                        if similarity >= 0.5:
                            feature_scores[a] -= sigmoid_weights[j] * diff * similarity  # HIT
                        else:
                            feature_scores[a] += sigmoid_weights[j] * diff * (1 - similarity)  # MISS


        return feature_scores

    def _run_algorithm(self):
        n_samples = self._datalen
        n_features = self._num_attributes
        total_sum = 0.0
        total_sum_sq = 0.0
        count = 0
        for i in range(n_samples):
            dists = np.array(self._distance_array[i], dtype=float)
            total_sum += dists.sum()
            total_sum_sq += (dists ** 2).sum()
            count += dists.size

        mean_dist = total_sum / count
        variance = (total_sum_sq / count) - (mean_dist ** 2)
        std_dist = np.sqrt(variance) if variance > 0 else 0.0
        nan_mask = np.isnan(self._X)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._score_instance)(i, mean_dist, std_dist, nan_mask)
            for i in range(n_samples)
        )
        feature_scores = np.sum(results, axis=0)
        total_weight = np.sum(np.abs(feature_scores))
        if total_weight > 0:
            feature_scores /= total_weight

        return feature_scores