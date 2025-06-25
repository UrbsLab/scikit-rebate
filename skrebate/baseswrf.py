# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from joblib import Parallel, delayed
from .relieff import ReliefF
import matplotlib.pyplot as plt


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

def swrf_weight(distances, mean, std, dead_band):
    return sigmoid_weight(distances, mean, std)


def tbd1_weight(distances, mean, std, dead_band):
    weights = np.zeros_like(distances)
    for i, d in enumerate(distances):
        if abs(d - mean) < dead_band:
            weights[i] = 0
        else:
            weights[i] = 1.0 if d < mean else -1.0
    return weights


def tbd2_weight(distances, mean, std, dead_band):
    weights = np.zeros_like(distances)
    for i, d in enumerate(distances):
        if abs(d - mean) < dead_band:
            weights[i] = 0
        elif d < mean:
            weights[i] = (mean - d) / (mean - (mean - 2 * dead_band))
        else:
            weights[i] = -(d - mean) / ((mean + 2 * dead_band) - mean)
    # weights is the z-score of the the dist
    weights = np.clip(weights, -1, 1)
    return weights


class BaseSWRF(ReliefF):
    def __init__(self, name, weight_func, ignore_far=False, **kwargs):
        super().__init__(**kwargs)
        self.weight_func = weight_func
        self.ignore_far = ignore_far
        self.name = name
        self.distance_weight_log = []  # For logging (distance, weight) pairs
        self.instance_dist_stats = []  # For * variants to approximate expected curve

    def _score_instance(self, inst_idx, dist_params, nan_mask):
        mean_dist, std_dist, dead_band = dist_params
        n_features = self._num_attributes
        dist_i = np.zeros(self._datalen, dtype=float)

        for j in range(self._datalen):
            if j == inst_idx:
                continue
            elif j < inst_idx:
                dist_i[j] = self._distance_array[inst_idx][j]
            else:
                dist_i[j] = self._distance_array[j][inst_idx]

        if 'TBD2' in self.name:
            mean_inst = np.mean(dist_i)
            std_inst = np.std(dist_i)
            dead_band_inst = std_inst / 2.0
            weights = self.weight_func(dist_i, mean_inst, std_inst, dead_band_inst)
            self.instance_dist_stats.append((mean_inst, std_inst, dead_band_inst))
        else:
            weights = self.weight_func(dist_i, mean_dist, std_dist, dead_band)
            weights[inst_idx] = 0.0
            self.instance_dist_stats.append((mean_dist, std_dist, dead_band))

        # Apply ignore_far logic
        if self.ignore_far:
            for i, d in enumerate(dist_i):
                if d > mean_dist:
                    weights[i] = 0.0

        # Log (distance, weight) pairs
        self.distance_weight_log.extend([
            (dist_i[j], weights[j])
            for j in range(self._datalen)
            if j != inst_idx
        ])

        feature_scores = np.zeros(n_features, dtype=float)
        for j in range(self._datalen):
            if j == inst_idx or weights[j] == 0:
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
                    if self._y[inst_idx] == self._y[j]:
                        feature_scores[a] -= weights[j] * diff
                    else:
                        feature_scores[a] += weights[j] * diff

                elif self._class_type == 'multiclass':
                    if self._y[inst_idx] == self._y[j]:
                        feature_scores[a] -= weights[j] * diff
                    else:
                        P_ci = self.mcmap.get(self._y[inst_idx], 0) / float(sum(self.mcmap.values()))
                        P_cj = self.mcmap.get(self._y[j], 0) / float(sum(self.mcmap.values()))
                        factor = P_cj / (1 - P_ci) if (1 - P_ci) > 0 else 0
                        feature_scores[a] += weights[j] * diff * factor

                else:  # continuous
                    if self._labels_std is not None and self._labels_std > 0:
                        response_diff = self._y[inst_idx] - self._y[j]
                        similarity = np.exp(-(response_diff ** 2) / (2 * (self._labels_std ** 2)))
                        if similarity >= 0.5:
                            feature_scores[a] -= weights[j] * diff * similarity
                        else:
                            feature_scores[a] += weights[j] * diff * (1 - similarity)

        return feature_scores

    def _run_algorithm(self):
        n_samples = self._datalen
        nan_mask = np.isnan(self._X)
        dists_flat = np.concatenate([np.array(row) for row in self._distance_array])
        mean_dist = dists_flat.mean()
        std_dist = dists_flat.std()
        dead_band = std_dist / 2 if 'TBD' in self.name else 0

        self.distance_weight_log = []  # Reset log before run
        self.instance_dist_stats = []

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._score_instance)(i, (mean_dist, std_dist, dead_band), nan_mask)
            for i in range(n_samples)
        )
        feature_scores = np.sum(results, axis=0)
        total_weight = np.sum(np.abs(feature_scores))
        if total_weight > 0:
            feature_scores /= total_weight

        return feature_scores

    def plot_distance_weight_map(self, save_fig=None, show_expected=True):
        """Visualize actual (distance, weight) pairs collected during Relief run."""
        if not self.distance_weight_log:
            print("No data logged yet. Run the algorithm first.")
            return

        distances, weights = zip(*self.distance_weight_log)
        plt.figure(figsize=(10, 6))
        plt.scatter(distances, weights, alpha=0.3, s=10, label='Observed')

        if show_expected:
            if self.name == 'TBD2':
                # Star variant → average per-instance mean/std
                means, stds, deadband = zip(*self.instance_dist_stats)
                mean_dist = np.mean(means)
                std_dist = np.mean(stds)
                dead_band = np.mean(deadband)
                # dead_band = std_dist / 4.0 if 'TBD' in self.name else 0
            else:
                x_vals = np.linspace(min(distances), max(distances), 500)
                mean_dist = np.mean(x_vals)
                std_dist = np.std(x_vals)
                dead_band = std_dist / 4.0 if 'TBD' in self.name else 0

            if 'SWRF' in self.name:
                y_vals = swrf_weight(x_vals, mean_dist, std_dist, dead_band)
            elif 'TBD_1' in self.name:
                y_vals = tbd1_weight(x_vals, mean_dist, std_dist, dead_band)
            elif 'TBD_2' in self.name:
                y_vals = tbd2_weight(x_vals, mean_dist, std_dist, dead_band)
            else:
                y_vals = None

            # Apply ignore_far logic
            if self.ignore_far:
                for i, d in enumerate(y_vals):
                    if d < mean_dist:
                        y_vals[i] = 0.0

            if y_vals is not None:
                plt.plot(x_vals, y_vals, label='Expected', linewidth=2, color='black')

        plt.title(f'Distance-to-Weight Mapping: {self.name}')
        plt.xlabel('Distance from Target Instance')
        plt.ylabel('Scoring Weight')
        plt.grid(True)
        plt.ylim(-1.1, 1.1)
        plt.legend()
        if save_fig:
            plt.savefig(save_fig)
        else:
            plt.show()


class SWRFstar2(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('SWRF', swrf_weight, ignore_far=False, **kwargs)


class SWRF(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('SWRF', swrf_weight, ignore_far=True, **kwargs)


class TBD1(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('TBD_1', tbd1_weight, ignore_far=True, **kwargs)


class TBD1star(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('TBD_1*', tbd1_weight, ignore_far=False, **kwargs)


class TBD2(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('TBD_2', tbd2_weight, ignore_far=True, **kwargs)


class TBD2star(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('TBD_2*', tbd2_weight, ignore_far=False, **kwargs)
