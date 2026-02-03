# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from joblib import Parallel, delayed
from .relieff import ReliefF
import matplotlib.pyplot as plt
import time
from pyinstrument import Profiler
from .scoring_utils import ramp_vec


# def sigmoid_weight(distances, mean_dist, std_dist):
#     # start_time_std_eff = time.time()
#     std_eff = std_dist if std_dist != 0 else 1.0
#     # end_time_std_eff = time.time()
#     # total_time_std_eff = end_time_std_eff - start_time_std_eff
#     # print("Time taken to create std eff:", total_time_std_eff, "seconds\n")

#     # start_time_diff = time.time()
#     diff = (distances - mean_dist) * (4.0 / std_eff)
#     # end_time_diff = time.time()
#     # total_time_diff = end_time_diff - start_time_diff
#     # print("Time taken to create diff:", total_time_diff, "seconds\n")

#     # start_time_clip = time.time()
#     diff = np.clip(diff, -50, 50)
#     # end_time_clip = time.time()
#     # total_time_clip = end_time_clip - start_time_clip
#     # print("Time taken to create clip:", total_time_clip, "seconds\n")

#     return 2.0 / (1.0 + np.exp(diff)) - 1.0

# def ramp_function(data_type, attr, fname, xinstfeature, xNNifeature):
#     diff = 0
#     mmdiff = attr[fname][3]
#     rawfd = abs(xinstfeature - xNNifeature)
#     if data_type == 'mixed':
#         standDev = attr[fname][4]
#         if rawfd > standDev:
#             diff = 1
#         else:
#             diff = rawfd / mmdiff
#     else:
#         diff = rawfd / mmdiff
#     return diff

# # def sigmoid_weight(distances, mean_dist, std_dist):
# #     diff = (distances - mean_dist) * (4.0 / std_dist)
# #     diff = np.clip(diff, -50, 50)
# #     return 2.0 / (1.0 + np.exp(diff)) - 1.0
# #
# #
# # def tbd1_weight(distances, mean, std, dead_band):
# #     weights = np.zeros_like(distances)
# #     for i, d in enumerate(distances):
# #         if abs(d - mean) < dead_band:
# #             weights[i] = 0
# #         else:
# #             weights[i] = 1.0 if d < mean else -1.0
# #     return weights


# # def tbd2_weight(distances, mean, std, dead_band):
# #     weights = np.zeros_like(distances)
# #     for i, d in enumerate(distances):
# #         if abs(d - mean) < dead_band:
# #             weights[i] = 0
# #         elif d < mean:
# #             weights[i] = (mean - d) / (mean - (mean - 2 * dead_band))
# #         else:
# #             weights[i] = -(d - mean) / ((mean + 2 * dead_band) - mean)
# #     # weights is the z-score of the the dist
# #     weights = np.clip(weights, -1, 1)
# #     return weights

# def deadband_bounds(mean, std, dead_band):
#     if dead_band is None or dead_band <= 0:
#         half = (std / 2.0) if std > 0 else 0.0
#     else:
#         half = float(dead_band)
#     lower = mean - half
#     upper = mean + half
#     return lower, upper

# def swrf_weight(distances, mean, std, dead_band=None):
#     return sigmoid_weight(distances, mean, std)

# def multiswrf_weight(distances, mean, std, dead_band=None):
#     distances = np.asarray(distances, dtype=float)
#     # start_time_db = time.time()
#     lower, upper = deadband_bounds(mean, std, dead_band)
#     # end_time_db = time.time()
#     # total_time_db = end_time_db - start_time_db
#     # print("Time taken to create deadband bounds:", total_time_db, "seconds\n")
#     w_sig = sigmoid_weight(distances, mean, std)
#     # weights = np.where(distances < lower,  1.0,
#     #            np.where(distances > upper, -1.0, w_sig))
#     weights = w_sig
#     # end_time_function = time.time()
#     # total_time_function = end_time_function - start_time_db
#     # print("Time taken from start of deadband bounds to before final return for multiswrf_weight:", total_time_function, "seconds\n")
#     return weights

# def multiswrfdb_weight(distances, mean, std, dead_band=None):
#     distances = np.asarray(distances, dtype=float)
#     lower, upper = deadband_bounds(mean, std, dead_band)
#     print("Lower (mean - dead_band):", lower, "\n")
    
#     weights = np.zeros_like(distances, dtype=float)
    
#     # scale factor for smoothness
#     scale = 4.0 / (std if std > 0 else 1.0)
    
#     near_mask = distances < lower
#     # weights[near_mask] = 1.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower)))
#     weights[near_mask] = 2.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower))) - 1.0
#     # print("Number of neighbors during this iteration:", near_mask.sum(), "\n")
    
#     far_mask = distances > upper
#     # weights[far_mask] = -1.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper)))
#     weights[far_mask] = -2.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper))) + 1.0
    
#     return weights

# def multiswrfdb_linear_weight(distances, mean, std, dead_band=None):
#     distances = np.asarray(distances, dtype=float)
#     lower, upper = deadband_bounds(mean, std, dead_band)
    
#     weights = np.zeros_like(distances, dtype=float)
    
#     near_mask = distances < lower
#     # linear weighting based on how many SD's an instance is from the lower bound; becomes 1 at 1.5 SD (b/c that is 2 SD away from mean)
#     weights[near_mask] = (((distances[near_mask] - lower) / std) / (1.5)) * -1
#     # make sure the weights don't exceed 1:
#     weights[near_mask] = np.minimum(weights[near_mask], 1.0)
    
#     far_mask = distances > upper
#     # weights between 0 and -1
#     weights[far_mask] = (((distances[far_mask] - upper) / std) / (1.5)) * -1
#     weights[far_mask] = np.maximum(weights[far_mask], -1.0)
    
#     return weights

# def multiswrfdb_exponential_weight(distances, mean, std, dead_band=None):
#     distances = np.asarray(distances, dtype=float)
#     lower, upper = deadband_bounds(mean, std, dead_band)
    
#     weights = np.zeros_like(distances, dtype=float)
    
#     near_mask = distances < lower
#     # weights[near_mask] = 1.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower)))
#     # weights[near_mask] = 2.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower))) - 1.0
#     # weights[near_mask] = (2.0 / (1.0 + np.exp(-scale*(distances[near_mask] - lower))) - 1.0) * -1
#     weights[near_mask] = ((distances[near_mask] - lower) / std)**2 / (1.5)**2
#     # make sure the weights don't exceed 1:
#     weights[near_mask] = np.minimum(weights[near_mask], 1.0)
    
#     far_mask = distances > upper
#     # weights[far_mask] = -1.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper)))
#     # weights[far_mask] = -2.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper))) + 1.0
#     weights[far_mask] = (((distances[far_mask] - upper) / std)**2 / (1.5)**2) * -1
#     # make sure the weights don't go lower than -1:
#     weights[far_mask] = np.maximum(weights[far_mask], -1.0)
    
#     return weights

# def multiswrfdb_linear_3sd_weight(distances, mean, std, dead_band=None):
#     distances = np.asarray(distances, dtype=float)
#     lower, upper = deadband_bounds(mean, std, dead_band)
    
#     weights = np.zeros_like(distances, dtype=float)
    
#     near_mask = distances < lower
#     # linear weighting based on how many SD's an instance is from the lower bound; becomes 1 at 1.5 SD (b/c that is 2 SD away from mean)
#     weights[near_mask] = (((distances[near_mask] - lower) / std) / (2.5)) * -1
#     # make sure the weights don't exceed 1:
#     weights[near_mask] = np.minimum(weights[near_mask], 1.0)
    
#     far_mask = distances > upper
#     # weights between 0 and -1
#     weights[far_mask] = (((distances[far_mask] - upper) / std) / (2.5)) * -1
#     weights[far_mask] = np.maximum(weights[far_mask], -1.0)
    
#     return weights

# def multiswrfdb_exponential_3sd_weight(distances, mean, std, dead_band=None):
#     distances = np.asarray(distances, dtype=float)
#     lower, upper = deadband_bounds(mean, std, dead_band)
    
#     weights = np.zeros_like(distances, dtype=float)
    
#     near_mask = distances < lower
#     # weights[near_mask] = 1.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower)))
#     # weights[near_mask] = 2.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower))) - 1.0
#     # weights[near_mask] = (2.0 / (1.0 + np.exp(-scale*(distances[near_mask] - lower))) - 1.0) * -1
#     weights[near_mask] = ((distances[near_mask] - lower) / std)**2 / (2.5)**2
#     # make sure the weights don't exceed 1:
#     weights[near_mask] = np.minimum(weights[near_mask], 1.0)
    
#     far_mask = distances > upper
#     # weights[far_mask] = -1.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper)))
#     # weights[far_mask] = -2.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper))) + 1.0
#     weights[far_mask] = (((distances[far_mask] - upper) / std)**2 / (2.5)**2) * -1
#     # make sure the weights don't go lower than -1:
#     weights[far_mask] = np.maximum(weights[far_mask], -1.0)
    
#     return weights

# class BaseSWRF(ReliefF):
#     def __init__(self, name, weight_func, ignore_far=False, **kwargs):
#         super().__init__(**kwargs)
#         self.weight_func = weight_func
#         self.ignore_far = ignore_far
#         self.name = name
#         self.distance_weight_log = []  # For logging (distance, weight) pairs
#         self.instance_dist_stats = []  # For * variants to approximate expected curve
#         # NEW: creating new instance variable to keep track of (distance, weight) pairs, but "distance" is instead STD away from target instance
#         self.std_weight_log = []

#     def _score_instance(self, inst_idx, dist_params, nan_mask):
#         mean_dist, std_dist, dead_band = dist_params
#         n_features = self._num_attributes
#         dist_i = np.zeros(self._datalen, dtype=float)

#         for j in range(self._datalen):
#             if j == inst_idx:
#                 continue
#             elif j < inst_idx:
#                 dist_i[j] = self._distance_array[inst_idx][j]
#             else:
#                 dist_i[j] = self._distance_array[j][inst_idx]

#         if 'MultiSWRF' in self.name:
#             # NEW: scaling distances so that max is 1; maximum distance present between target instance & any other instance
#             # max_dist = np.max(dist_i)
#             # max_dist = np.max(self._distance_array)
#             # dist_i = dist_i / max_dist

#             mean_inst = np.mean(dist_i)
#             # print("Mean for this target instance in MultiSWRFDB:", mean_inst)
#             std_inst = np.std(dist_i)
#             dead_band_inst = std_inst / 2.0
#             # print("Mean calculated in score_instance for current target:", mean_inst, "\n")
#             # print("Deadband calculated in score_instance (STD / 2) for current target:", dead_band_inst, "\n")
#             weights = self.weight_func(dist_i, mean_inst, std_inst, dead_band_inst)
#             # start_time = time.time()
#             # self.instance_dist_stats.append((mean_inst, std_inst, dead_band_inst))
#             # print("Time taken to append to instance_dist_stats:", time.time() - start_time, "seconds\n")

#             # NEW: copy of dist_i where distances are translated to STD
#             dist_i_std =  (dist_i - mean_inst) / std_inst
#         else:
#             # NEW: scaling distances so that max is 1; maximum distance present between any 2 instances in the dataset
#             # max_dist = np.max(self._distance_array)
#             # dist_i = dist_i / max_dist
#             # mean_dist = mean_dist / max_dist
#             # std_dist = std_dist / max_dist
#             # dead_band = dead_band / max_dist

#             mean_inst = mean_dist
#             weights = self.weight_func(dist_i, mean_dist, std_dist, dead_band)
#             weights[inst_idx] = 0.0
#             # self.instance_dist_stats.append((mean_dist, std_dist, dead_band))

#             # NEW: copy of dist_i where distances are translated to STD
#             dist_i_std =  (dist_i - mean_dist) / std_dist

#         # Apply ignore_far logic
#         if self.ignore_far:
#             for i, d in enumerate(dist_i):
#                 if d > mean_inst:
#                     weights[i] = 0.0

#         # # Log (distance, weight) pairs
#         # self.distance_weight_log.extend([
#         #     (dist_i[j], weights[j])
#         #     for j in range(self._datalen)
#         #     if j != inst_idx
#         # ])

#         # # NEW: Log (distance, weight) pairs for STD
#         # self.std_weight_log.extend([
#         #     (dist_i_std[j], weights[j])
#         #     for j in range(self._datalen)
#         #     if j != inst_idx
#         # ])

#         feature_scores = np.zeros(n_features, dtype=float)
#         for j in range(self._datalen):
#             if j == inst_idx or weights[j] == 0:
#                 continue
#             for a in range(n_features):
#                 if nan_mask[inst_idx][a] or nan_mask[j][a]:
#                     continue
#                 ftype = self.attr[self._headers[a]][0]
#                 if ftype == 'continuous':
#                     diff = ramp_function(self.data_type, self.attr, self._headers[a],
#                                          self._X[inst_idx][a], self._X[j][a])
#                 else:
#                     diff = float(self._X[inst_idx][a] != self._X[j][a])

#                 if self._class_type == 'binary':
#                     if self._y[inst_idx] == self._y[j]:
#                         feature_scores[a] -= weights[j] * diff
#                     else:
#                         feature_scores[a] += weights[j] * diff

#                 elif self._class_type == 'multiclass':
#                     if self._y[inst_idx] == self._y[j]:
#                         feature_scores[a] -= weights[j] * diff
#                     else:
#                         P_ci = self.mcmap.get(self._y[inst_idx], 0) / float(sum(self.mcmap.values()))
#                         P_cj = self.mcmap.get(self._y[j], 0) / float(sum(self.mcmap.values()))
#                         factor = P_cj / (1 - P_ci) if (1 - P_ci) > 0 else 0
#                         feature_scores[a] += weights[j] * diff * factor

#                 else:  # continuous
#                     if self._labels_std is not None and self._labels_std > 0:
#                         response_diff = self._y[inst_idx] - self._y[j]
#                         similarity = np.exp(-(response_diff ** 2) / (2 * (self._labels_std ** 2)))
#                         if similarity >= 0.5:
#                             feature_scores[a] -= weights[j] * diff * similarity
#                         else:
#                             feature_scores[a] += weights[j] * diff * (1 - similarity)

#         return feature_scores

#     def _run_algorithm(self):
#         # start_time = time.time()
#         n_samples = self._datalen
#         # print("Time taken to create n_samples:", time.time() - start_time, "seconds\n")
#         nan_mask = np.isnan(self._X)
#         # print("Time taken to create nan_mask:", time.time() - start_time, "seconds\n")
#         dists_flat = np.concatenate([np.array(row) for row in self._distance_array])
#         # print("Time taken to create dists_flat:", time.time() - start_time, "seconds\n")
#         mean_dist = dists_flat.mean()
#         # print("Time taken to create mean_dist:", time.time() - start_time, "seconds\n")
#         # print("Mean dist:", mean_dist, "\n")
#         std_dist = dists_flat.std()
#         # print("Time taken to create std_dist:", time.time() - start_time, "seconds\n")
#         # print("STD of the distances:", std_dist, "\n")
#         dead_band = std_dist / 2 if 'MultiSWRF' in self.name else 0
#         # print("Time taken to create dead_band:", time.time() - start_time, "seconds\n")
#         # if dead_band != 0:
#         #     print("'If MultiSWRF in self.name' check is True")
#         # else:
#         #     print("'If MultiSWRF in self.name' check is False")
#         # print("Dead band (STD / 2):", dead_band, "\n")

#         self.distance_weight_log = []  # Reset log before run
#         # print("Time taken to empty distance_weight_log:", time.time() - start_time, "seconds\n")
#         self.instance_dist_stats = []
#         # print("Time taken to empty instance_dist_stats:", time.time() - start_time, "seconds\n")
#         # NEW: also reset this:
#         self.std_weight_log = []
#         # print("Time taken to empty std_weight_log:", time.time() - start_time, "seconds\n")

#         profiler = Profiler()
#         profiler.start()
#         results = Parallel(n_jobs=self.n_jobs)(
#             delayed(self._score_instance)(i, (mean_dist, std_dist, dead_band), nan_mask)
#             for i in range(n_samples)
#         )
#         profiler.stop()
#         profiler.print()
        
#         # print("Time taken to finish scoring instances:", time.time() - start_time, "seconds\n")
#         feature_scores = np.sum(results, axis=0)
#         # print("Time taken to sum feature scores:", time.time() - start_time, "seconds\n")
#         total_weight = np.sum(np.abs(feature_scores))
#         # print("Time taken to sum total_weight:", time.time() - start_time, "seconds\n")
#         if total_weight > 0:
#             feature_scores /= total_weight
#         # print("Time taken to divide feature scores by total_weight:", time.time() - start_time, "seconds\n")
#         # print("Feature scores:", feature_scores, "\n")
#         return feature_scores

#     def plot_distance_weight_map(self, save_fig=None, show_expected=True):
#         """Visualize actual (distance, weight) pairs collected during Relief run."""
#         if not self.distance_weight_log:
#             print("No data logged yet. Run the algorithm first.")
#             return

#         distances, weights = zip(*self.distance_weight_log)
#         # NEW: use self.std_weight_log for plotting
#         distances_std, weights_std = zip(*self.std_weight_log)
#         plt.figure(figsize=(10, 6))
#         # plt.scatter(distances, weights, alpha=0.3, s=10, label='Observed')
#         plt.scatter(distances_std, weights_std, alpha=0.3, s=10, label='Observed')

#         if show_expected:
#             if 'MultiSWRF' in self.name: 
#                 # Star variant → average per-instance mean/std
#                 x_vals = np.linspace(min(distances), max(distances), 500)
#                 means, stds, deadband = zip(*self.instance_dist_stats)
#                 mean_dist = np.mean(means)
#                 std_dist = np.mean(stds)
#                 dead_band = np.mean(deadband)
#                 # dead_band = std_dist / 4.0 if 'TBD' in self.name else 0

#                 # NEW: for plotting in terms of STD
#                 x_vals_std = (x_vals - mean_dist) / std_dist
#             else:
#                 x_vals = np.linspace(min(distances), max(distances), 500)
#                 mean_dist = np.mean(distances)
#                 std_dist = np.std(distances)
#                 dead_band = std_dist / 4.0 if 'MultiSWRF' in self.name else 0

#                 # NEW: for plotting in terms of STD
#                 x_vals_std = (x_vals - mean_dist) / std_dist

#             if self.name in ('SWRF', 'SWRF*'):
#                 y_vals = swrf_weight(x_vals, mean_dist, std_dist, dead_band)
#             elif self.name in ('MultiSWRF', 'MultiSWRF*'):
#                 y_vals = multiswrf_weight(x_vals, mean_dist, std_dist, dead_band)
#             elif 'MultiSWRFDB' in self.name:
#                 if 'linear' in self.name:
#                     if '3SD' in self.name:
#                         y_vals = multiswrfdb_linear_3sd_weight(x_vals, mean_dist, std_dist, dead_band)
#                     else:
#                         y_vals = multiswrfdb_linear_weight(x_vals, mean_dist, std_dist, dead_band)
#                 elif 'exponential' in self.name:
#                     if '3SD' in self.name:
#                         y_vals = multiswrfdb_exponential_3sd_weight(x_vals, mean_dist, std_dist, dead_band)
#                     else:
#                         y_vals = multiswrfdb_exponential_weight(x_vals, mean_dist, std_dist, dead_band)
#                 else:
#                     y_vals = multiswrfdb_weight(x_vals, mean_dist, std_dist, dead_band)
#             else:
#                 y_vals = None

#             # Apply ignore_far logic
#             if self.ignore_far:
#                 for i, d in enumerate(x_vals):
#                     if d > mean_dist:
#                         y_vals[i] = 0.0

#             if y_vals is not None:
#                 # plt.plot(x_vals, y_vals, label='Expected', linewidth=2, color='black')
#                 # NEW: use x_vals STD instead
#                 plt.plot(x_vals_std, y_vals, label='Expected', linewidth=2, color='black')

#         plt.title(f'Distance-to-Weight Mapping: {self.name}')
#         plt.xlabel('Distance from Target Instance')
#         # plt.xlabel('Standard Deviations (Distance) from Target Instance')
#         plt.ylabel('Scoring Weight')
#         plt.grid(True)
#         # NEW: grid lines different from x-tick labels:
#         plt.gca().set_xticks(np.arange(-3, 4, 1), minor=False)
#         plt.ylim(-1.1, 1.1)
#         # NEW: xlim to set x-axis values between 0 and 1.0 for all graphs (consistent)
#         # plt.xlim(0, 1.0)
#         plt.xlim(-3.0, 3.0)
#         # plt.xticks(np.linspace(0, 1.0, num=6))
#         # plt.xticks([-3, -2, -1, 0, 1, 2, 3])
#         plt.xticks(
#             [-0.5, 0, 0.5],
#             ['(μ - σ/2)', 'μ', '(μ + σ/2)']
#         )
#         # NEW: dotted lines for deadband zone boundaries (0.5 SD on either side of mean)
#         plt.axvline(x=-0.5, color='red', linestyle='dotted')
#         plt.axvline(x=0.5,  color='red', linestyle='dotted')
#         plt.legend()
#         if save_fig:
#             plt.savefig(save_fig)
#         else:
#             plt.show()

# *** REFACTORED VERSION; KEEPING IT CONSISTENT WITH TRADITIONAL RBAs

def sigmoid_weight(distances, mean_dist, std_dist):
    # start_time_std_eff = time.time()
    std_eff = std_dist if std_dist != 0 else 1.0
    # end_time_std_eff = time.time()
    # total_time_std_eff = end_time_std_eff - start_time_std_eff
    # print("Time taken to create std eff:", total_time_std_eff, "seconds\n")

    # start_time_diff = time.time()
    # diff = (distances - mean_dist) * (4.0 / std_eff)
    # *** all weights "positive" with refactored function
    diff = -(np.abs(distances - mean_dist)) * (4.0 / std_eff)
    # end_time_diff = time.time()
    # total_time_diff = end_time_diff - start_time_diff
    # print("Time taken to create diff:", total_time_diff, "seconds\n")

    # start_time_clip = time.time()
    diff = np.clip(diff, -50, 50)
    # end_time_clip = time.time()
    # total_time_clip = end_time_clip - start_time_clip
    # print("Time taken to create clip:", total_time_clip, "seconds\n")

    return 2.0 / (1.0 + np.exp(diff)) - 1.0

def swrf_weight(distances, std, mean):
    return sigmoid_weight(distances, mean, std)

def multiswrf_weight(distances, std, mean):
    distances = np.asarray(distances, dtype=float)
    # start_time_db = time.time()
    # lower, upper = deadband_bounds(mean, std, dead_band)
    # end_time_db = time.time()
    # total_time_db = end_time_db - start_time_db
    # print("Time taken to create deadband bounds:", total_time_db, "seconds\n")
    w_sig = sigmoid_weight(distances, mean, std)
    # weights = np.where(distances < lower,  1.0,
    #            np.where(distances > upper, -1.0, w_sig))
    weights = w_sig
    # end_time_function = time.time()
    # total_time_function = end_time_function - start_time_db
    # print("Time taken from start of deadband bounds to before final return for multiswrf_weight:", total_time_function, "seconds\n")
    return weights

def multiswrfdb_weight(distances, std, bound):
    distances = np.asarray(distances, dtype=float)
    
    weights = np.zeros_like(distances, dtype=float)
    
    # scale factor for smoothness
    scale = 4.0 / (std if std > 0 else 1.0)
    
    # if bound_type == "lower":
    #     # weights[near_mask] = 1.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower)))
    #     weights = 2.0 / (1.0 + np.exp(scale*(distances - bound))) - 1.0
    #     # print("Number of neighbors during this iteration:", near_mask.sum(), "\n")
    # else:
    #     # weights[far_mask] = -1.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper)))
    #     weights = -2.0 / (1.0 + np.exp(-scale*(distances - bound))) + 1.0
    weights = 2.0 / (1.0 + np.exp(scale*(-np.abs(distances - bound)))) - 1.0
    
    return weights

def multiswrfdb_linear_weight(distances, std, bound):
    distances = np.asarray(distances, dtype=float)
    
    weights = np.zeros_like(distances, dtype=float)
    
    # if bound_type == "lower":
    #     # linear weighting based on how many SD's an instance is from the lower bound; becomes 1 at 1.5 SD (b/c that is 2 SD away from mean)
    #     weights = (((distances - bound) / std) / (1.5)) * -1
    #     # make sure the weights don't exceed 1:
    #     weights = np.minimum(weights, 1.0)
    # else:
    #     # weights between 0 and -1
    #     weights = (((distances - bound) / std) / (1.5)) * -1
    #     weights = np.maximum(weights, -1.0)
    # weights = ((-np.abs(distances - bound) / std) / (1.5)) * -1

    # linear weighting based on how many SD's an instance is from the lower bound; becomes 1 at 1.5 SD (b/c that is 2 SD away from mean)
    weights = ((np.abs(distances - bound) / std) / (1.5))
    # make sure the weights don't exceed 1:
    weights = np.minimum(weights, 1.0)
    
    return weights

def multiswrfdb_exponential_weight(distances, std, bound):
    distances = np.asarray(distances, dtype=float)
    
    weights = np.zeros_like(distances, dtype=float)
    
    # if bound_type == "lower":
    #     # weights[near_mask] = 1.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower)))
    #     # weights[near_mask] = 2.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower))) - 1.0
    #     # weights[near_mask] = (2.0 / (1.0 + np.exp(-scale*(distances[near_mask] - lower))) - 1.0) * -1
    #     weights = ((distances - bound) / std)**2 / (1.5)**2
    #     # make sure the weights don't exceed 1:
    #     weights = np.minimum(weights, 1.0)
    # else:
    #     # weights[far_mask] = -1.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper)))
    #     # weights[far_mask] = -2.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper))) + 1.0
    #     weights = (((distances - bound) / std)**2 / (1.5)**2) * -1
    #     # make sure the weights don't go lower than -1:
    #     weights = np.maximum(weights, -1.0)
    weights = ((distances - bound) / std)**2 / (1.5)**2
    # make sure the weights don't exceed 1:
    weights = np.minimum(weights, 1.0)
    
    return weights

def multiswrfdb_linear_3sd_weight(distances, std, bound):
    distances = np.asarray(distances, dtype=float)
    
    weights = np.zeros_like(distances, dtype=float)
    
    # near_mask = distances < lower
    # # linear weighting based on how many SD's an instance is from the lower bound; becomes 1 at 1.5 SD (b/c that is 2 SD away from mean)
    # weights[near_mask] = (((distances[near_mask] - lower) / std) / (2.5)) * -1
    # # make sure the weights don't exceed 1:
    # weights[near_mask] = np.minimum(weights[near_mask], 1.0)
    
    # far_mask = distances > upper
    # # weights between 0 and -1
    # weights[far_mask] = (((distances[far_mask] - upper) / std) / (2.5)) * -1
    # weights[far_mask] = np.maximum(weights[far_mask], -1.0)

    # linear weighting based on how many SD's an instance is from the lower bound; becomes 1 at 1.5 SD (b/c that is 2 SD away from mean)
    weights = ((np.abs(distances - bound) / std) / (2.5))
    # make sure the weights don't exceed 1:
    weights = np.minimum(weights, 1.0)
    
    return weights

def multiswrfdb_exponential_3sd_weight(distances, std, bound):
    distances = np.asarray(distances, dtype=float)
    
    weights = np.zeros_like(distances, dtype=float)
    
    # near_mask = distances < lower
    # # weights[near_mask] = 1.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower)))
    # # weights[near_mask] = 2.0 / (1.0 + np.exp(scale*(distances[near_mask] - lower))) - 1.0
    # # weights[near_mask] = (2.0 / (1.0 + np.exp(-scale*(distances[near_mask] - lower))) - 1.0) * -1
    # weights[near_mask] = ((distances[near_mask] - lower) / std)**2 / (2.5)**2
    # # make sure the weights don't exceed 1:
    # weights[near_mask] = np.minimum(weights[near_mask], 1.0)
    
    # far_mask = distances > upper
    # # weights[far_mask] = -1.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper)))
    # # weights[far_mask] = -2.0 / (1.0 + np.exp(-scale*(distances[far_mask] - upper))) + 1.0
    # weights[far_mask] = (((distances[far_mask] - upper) / std)**2 / (2.5)**2) * -1
    # # make sure the weights don't go lower than -1:
    # weights[far_mask] = np.maximum(weights[far_mask], -1.0)

    weights = ((distances - bound) / std)**2 / (2.5)**2
    # make sure the weights don't exceed 1:
    weights = np.minimum(weights, 1.0)
    
    return weights

class BaseSWRF(ReliefF):
    def __init__(self, name, weight_func, ignore_far=False, **kwargs):
        super().__init__(**kwargs)
        self.weight_func = weight_func
        self.ignore_far = ignore_far
        self.name = name
        self.distance_weight_log = []  # For logging (distance, weight) pairs
        self.instance_dist_stats = []  # For * variants to approximate expected curve
        # NEW: creating new instance variable to keep track of (distance, weight) pairs, but "distance" is instead STD away from target instance
        self.std_weight_log = []

    def _find_neighbors(self, inst, global_mean_dist, global_std_dist):
        if 'Multi' in self.name: # calculate mean and std for each target instance
            dist_vect = []
            for j in range(self._datalen): # compute dist_vect for current target instance
                if inst != j:
                    locator = [inst, j]
                    if inst < j:
                        locator.reverse()
                    dist_vect.append(self._distance_array[locator[0]][locator[1]])

            dist_vect = np.array(dist_vect)
            inst_avg_dist = np.average(dist_vect)
            true_std = np.std(dist_vect)
            
            if 'MultiSWRFDB' in self.name:
                inst_deadband = np.std(dist_vect) / 2. # calculate deadband distance

                # NEW: unique mean, std, and deadband values for this target instance, used to construct the expected curve
                # self.instance_dist_stats.append((inst_avg_dist, true_std, inst_std))

                # Defining a narrower radius based on the average instance distance minus the standard deviation of instance distances.
                near_threshold = inst_avg_dist - inst_deadband

                if self.ignore_far: # only near neighbors
                    NN_near = []
                    distances_NN_near = []
                    for j in range(self._datalen):
                        if inst != j:
                            locator = [inst, j]
                            if inst < j:
                                locator.reverse()
                            # NEW: how many standard deviations d is from the current mean distance
                            d = self._distance_array[locator[0]][locator[1]]
                            # std_d = (d - inst_avg_dist) / true_std
                            if d < near_threshold:
                                NN_near.append(j)
                                distances_NN_near.append(d)
                            #     # NEW: for plotting
                            #     self.distance_weight_log.append((d, 1.0))
                            #     self.std_weight_log.append((std_d, 1.0))
                            # else:
                            #     self.distance_weight_log.append((d, 0.0))
                            #     self.std_weight_log.append((std_d, 0.0))
                    
                    NN_near = np.array(NN_near)
                    # weights_near = self.weight_func(distances_NN_near, true_std, near_threshold, bound_type="lower")
                    weights_near = self.weight_func(distances_NN_near, true_std, near_threshold)

                    self.instance_dist_stats.append((inst_avg_dist, true_std, inst_deadband))
                    self.distance_weight_log.extend(zip(distances_NN_near, weights_near))
                    distances_NN_near_std = (np.array(distances_NN_near) - inst_avg_dist) / true_std
                    self.std_weight_log.extend(zip(distances_NN_near_std, weights_near))

                    # if ignore_far, only return near neighbors and near weights
                    return NN_near, weights_near, None, None
                else: # far neighbors included (for star algorithms)
                    far_threshold = inst_avg_dist + inst_deadband
                    NN_near = []
                    NN_far = []
                    distances_NN_near = []
                    distances_NN_far = []
                    for j in range(self._datalen):
                        if inst != j:
                            locator = [inst, j]
                            if inst < j:
                                locator.reverse()
                            # NEW: how many standard deviations d is from the current mean distance
                            d = self._distance_array[locator[0]][locator[1]]
                            std_d = (d - inst_avg_dist) / true_std

                            if d < near_threshold:
                                NN_near.append(j)
                                distances_NN_near.append(d)
                                # # NEW: for plotting
                                # self.distance_weight_log.append((d, 1.0))
                                # self.std_weight_log.append((std_d, 1.0))
                            elif d > far_threshold:
                                NN_far.append(j)
                                distances_NN_far.append(d)
                            #     # NEW: for plotting
                            #     self.distance_weight_log.append((d, -1.0))
                            #     self.std_weight_log.append((std_d, -1.0))
                            # else:
                            #     # NEW: for plotting
                            #     self.distance_weight_log.append((d, 0.0))
                            #     self.std_weight_log.append((std_d, 0.0))
                    
                    NN_near = np.array(NN_near)
                    NN_far = np.array(NN_far)
                    # weights_near = self.weight_func(distances_NN_near, true_std, near_threshold, bound_type="lower")
                    # weights_far = self.weight_func(distances_NN_far, true_std, far_threshold, bound_type="upper")
                    weights_near = self.weight_func(distances_NN_near, true_std, near_threshold)
                    weights_far = self.weight_func(distances_NN_far, true_std, far_threshold)

                    self.instance_dist_stats.append((inst_avg_dist, true_std, inst_deadband))
                    # Log (distance, weight) pairs
                    # self.distance_weight_log.extend([
                    #     (distances_NN_near[j], weights_near[j])
                    #     for j in range(len(weights_near))
                    # ])
                    # self.distance_weight_log.extend([
                    #     (distances_NN_far[j], weights_far[j])
                    #     for j in range(len(weights_far))
                    # ])
                    self.distance_weight_log.extend(zip(distances_NN_near, weights_near))
                    self.distance_weight_log.extend(zip(distances_NN_far, -1 * weights_far))

                    # NEW: Log (distance, weight) pairs for STD
                    # self.std_weight_log.extend([
                    #     (dist_i_std[j], weights[j])
                    #     for j in range(self._datalen)
                    #     if j != inst_idx
                    # ])
                    distances_NN_near_std = (np.array(distances_NN_near) - inst_avg_dist) / true_std
                    distances_NN_far_std = (np.array(distances_NN_far) - inst_avg_dist) / true_std
                    self.std_weight_log.extend(zip(distances_NN_near_std, weights_near))
                    self.std_weight_log.extend(zip(distances_NN_far_std, -1 * weights_far))
                    
                    # if ignore_far=False, return both near and far neighbors
                    return NN_near, weights_near, NN_far, weights_far
            else: # MultiSWRF
                if self.ignore_far: # only near neighbors
                    NN_near = []
                    distances_NN_near = []
                    for j in range(self._datalen):
                        if inst != j:
                            locator = [inst, j]
                            if inst < j:
                                locator.reverse()
                            # NEW: how many standard deviations d is from the current mean distance
                            d = self._distance_array[locator[0]][locator[1]]
                            # std_d = (d - inst_avg_dist) / true_std
                            if d < inst_avg_dist:
                                NN_near.append(j)
                                distances_NN_near.append(d)
                            #     # NEW: for plotting
                            #     self.distance_weight_log.append((d, 1.0))
                            #     self.std_weight_log.append((std_d, 1.0))
                            # else:
                            #     self.distance_weight_log.append((d, 0.0))
                            #     self.std_weight_log.append((std_d, 0.0))
                    
                    NN_near = np.array(NN_near)
                    weights_near = self.weight_func(distances_NN_near, true_std, inst_avg_dist)

                    self.instance_dist_stats.append((inst_avg_dist, true_std, None))
                    self.distance_weight_log.extend(zip(distances_NN_near, weights_near))
                    distances_NN_near_std = (np.array(distances_NN_near) - inst_avg_dist) / true_std
                    self.std_weight_log.extend(zip(distances_NN_near_std, weights_near))

                    # if ignore_far, only return near neighbors and near weights
                    return NN_near, weights_near, None, None
                else: # far neighbors included (for star algorithms)
                    NN_near = []
                    NN_far = []
                    distances_NN_near = []
                    distances_NN_far = []
                    for j in range(self._datalen):
                        if inst != j:
                            locator = [inst, j]
                            if inst < j:
                                locator.reverse()
                            # NEW: how many standard deviations d is from the current mean distance
                            d = self._distance_array[locator[0]][locator[1]]
                            std_d = (d - inst_avg_dist) / true_std

                            if d < inst_avg_dist:
                                NN_near.append(j)
                                distances_NN_near.append(d)
                                # # NEW: for plotting
                                # self.distance_weight_log.append((d, 1.0))
                                # self.std_weight_log.append((std_d, 1.0))
                            elif d > inst_avg_dist:
                                NN_far.append(j)
                                distances_NN_far.append(d)
                            #     # NEW: for plotting
                            #     self.distance_weight_log.append((d, -1.0))
                            #     self.std_weight_log.append((std_d, -1.0))
                            # else:
                            #     # NEW: for plotting
                            #     self.distance_weight_log.append((d, 0.0))
                            #     self.std_weight_log.append((std_d, 0.0))
                    
                    NN_near = np.array(NN_near)
                    NN_far = np.array(NN_far)
                    weights_near = self.weight_func(distances_NN_near, true_std, inst_avg_dist)
                    weights_far = self.weight_func(distances_NN_far, true_std, inst_avg_dist)

                    self.instance_dist_stats.append((inst_avg_dist, true_std, None))
                    self.distance_weight_log.extend(zip(distances_NN_near, weights_near))
                    self.distance_weight_log.extend(zip(distances_NN_far, -1 * weights_far))
                    distances_NN_near_std = (np.array(distances_NN_near) - inst_avg_dist) / true_std
                    distances_NN_far_std = (np.array(distances_NN_far) - inst_avg_dist) / true_std
                    self.std_weight_log.extend(zip(distances_NN_near_std, weights_near))
                    self.std_weight_log.extend(zip(distances_NN_far_std, -1 * weights_far))

                    # if ignore_far=False, return both near and far neighbors
                    return NN_near, weights_near, NN_far, weights_far
        else: # SWRF (use global mean and STD)
            if self.ignore_far: # only near neighbors
                NN_near = []
                distances_NN_near = []
                for j in range(self._datalen):
                    if inst != j:
                        locator = [inst, j]
                        if inst < j:
                            locator.reverse()
                        # NEW: how many standard deviations d is from the current mean distance
                        d = self._distance_array[locator[0]][locator[1]]
                        # std_d = (d - global_mean_dist) / global_std_dist
                        if d < global_mean_dist:
                            NN_near.append(j)
                            distances_NN_near.append(d)
                        #     # NEW: for plotting
                        #     self.distance_weight_log.append((d, 1.0))
                        #     self.std_weight_log.append((std_d, 1.0))
                        # else:
                        #     self.distance_weight_log.append((d, 0.0))
                        #     self.std_weight_log.append((std_d, 0.0))
                    
                NN_near = np.array(NN_near)
                weights_near = self.weight_func(distances_NN_near, global_std_dist, global_mean_dist)

                self.instance_dist_stats.append((global_mean_dist, global_std_dist, None))
                self.distance_weight_log.extend(zip(distances_NN_near, weights_near))
                distances_NN_near_std = (np.array(distances_NN_near) - global_mean_dist) / global_std_dist
                self.std_weight_log.extend(zip(distances_NN_near_std, weights_near))

                # if ignore_far, only return near neighbors and near weights
                return NN_near, weights_near, None, None
            else: # far neighbors included (for star algorithms)
                NN_near = []
                NN_far = []
                distances_NN_near = []
                distances_NN_far = []
                for j in range(self._datalen):
                    if inst != j:
                        locator = [inst, j]
                        if inst < j:
                            locator.reverse()
                        # NEW: how many standard deviations d is from the current mean distance
                        d = self._distance_array[locator[0]][locator[1]]
                        std_d = (d - global_mean_dist) / global_std_dist

                        if d < global_mean_dist:
                            NN_near.append(j)
                            distances_NN_near.append(d)
                            # # NEW: for plotting
                            # self.distance_weight_log.append((d, 1.0))
                            # self.std_weight_log.append((std_d, 1.0))
                        elif d > global_mean_dist:
                            NN_far.append(j)
                            distances_NN_far.append(d)
                        #     # NEW: for plotting
                        #     self.distance_weight_log.append((d, -1.0))
                        #     self.std_weight_log.append((std_d, -1.0))
                        # else:
                        #     # NEW: for plotting
                        #     self.distance_weight_log.append((d, 0.0))
                        #     self.std_weight_log.append((std_d, 0.0))
                    
                NN_near = np.array(NN_near)
                NN_far = np.array(NN_far)
                weights_near = self.weight_func(distances_NN_near, global_std_dist, global_mean_dist)
                weights_far = self.weight_func(distances_NN_far, global_std_dist, global_mean_dist)

                self.instance_dist_stats.append((global_mean_dist, global_std_dist, None))
                self.distance_weight_log.extend(zip(distances_NN_near, weights_near))
                self.distance_weight_log.extend(zip(distances_NN_far, -1 * weights_far))
                distances_NN_near_std = (np.array(distances_NN_near) - global_mean_dist) / global_std_dist
                distances_NN_far_std = (np.array(distances_NN_far) - global_mean_dist) / global_std_dist
                self.std_weight_log.extend(zip(distances_NN_near_std, weights_near))
                self.std_weight_log.extend(zip(distances_NN_far_std, -1 * weights_far))

                # if ignore_far=False, return both near and far neighbors
                return NN_near, weights_near, NN_far, weights_far
    
    # Same as compute_score used by MultiSURF, SURF, ReliefF, but instance weights are integrated here
    @staticmethod
    def compute_score(attr, mcmap, NN, feature, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, weights_inst, near=True):
        """Flexible feature scoring method that can be used with any core Relief-based method. Scoring proceeds differently
        based on whether endpoint is binary, multiclass, or continuous. This method is called for a single target instance
        + feature combination and runs over all items in NN. """

        fname = headers[feature]  # feature identifier
        ftype = attr[fname][0]  # feature type
        ctype = class_type  # class type (binary, multiclass, continuous)
        diff_hit = diff_miss = 0.0  # Tracks the score contribution
        # Tracks the number of hits/misses. Used in normalizing scores by 'k' in ReliefF, and by m or h in SURF, SURF*, MultiSURF*, and MultiSURF
        count_hit = count_miss = 0.0
        # Initialize 'diff' (The score contribution for this target instance and feature over all NN)
        diff = 0
        # mmdiff = attr[fname][3] # Max/Min range of values for target feature

        datalen = float(len(X))

        # If target instance is missing, then a 'neutral' score contribution of 0 is returned immediately since all NN comparisons will be against this missing value.
        if nan_entries[inst][feature]:
            return 0.
        # Note missing data normalization below regarding missing NN feature values is accomplished by counting hits and misses (missing values are not counted) (happens in parallel with hit/miss imbalance normalization)

        xinstfeature = X[inst][feature]  # value of target instances target feature.

        # NEW: bringing y_inst out of the loop and just looking it up once
        y_inst = y[inst]
        #--------------------------------------------------------------------------
        if ctype == 'binary':
            # for i in range(len(NN)):
            #     if nan_entries[NN[i]][feature]:  # skip any NN with a missing value for this feature.
            #         continue

            #     xNNifeature = X[NN[i]][feature]

            #     if near:  # SCORING FOR NEAR INSTANCES
            #         if y[inst] == y[NN[i]]:   # HIT
            #             count_hit += 1
            #             if ftype == 'continuous':
            #                 # diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff #Normalize absolute value of feature value difference by max-min value range for feature (so score update lies between 0 and 1)
            #                 diff_hit -= ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)
            #             else:  # discrete feature
            #                 if xinstfeature != xNNifeature:  # A difference in feature value is observed
            #                     # Feature score is reduced when we observe feature difference between 'near' instances with the same class.
            #                     diff_hit -= 1
            #         else:  # MISS
            #             count_miss += 1
            #             if ftype == 'continuous':
            #                 #diff_miss += abs(xinstfeature - xNNifeature) / mmdiff
            #                 diff_miss += ramp_function(data_type, attr, fname,
            #                                            xinstfeature, xNNifeature)
            #             else:  # discrete feature
            #                 if xinstfeature != xNNifeature:  # A difference in feature value is observed
            #                     # Feature score is increase when we observe feature difference between 'near' instances with different class values.
            #                     diff_miss += 1
                # else:  # SCORING FOR FAR INSTANCES (ONLY USED BY MULTISURF* BASED ON HOW CODED)
                #     if y[inst] == y[NN[i]]:   # HIT
                #         count_hit += 1
                #         if ftype == 'continuous':

                #             #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff  #Hits differently add continuous value differences rather than subtract them 
                #             diff_hit -= (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                #         else: #discrete feature
                #             if xinstfeature == xNNifeature: # The same feature value is observed (Used for more efficient 'far' scoring, since there should be fewer same values for 'far' instances)
                #                 diff_hit -= 1 # Feature score is reduced when we observe the same feature value between 'far' instances with the same class.
                #     else:  # MISS
                #         count_miss += 1
                #         if ftype == 'continuous':
                #             #diff_miss += abs(xinstfeature - xNNifeature) / mmdiff #Misses differntly subtract continuous value differences rather than add them 
                #             diff_miss += (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                #         else: #discrete feature
                #             if xinstfeature == xNNifeature: # The same feature value is observed (Used for more efficient 'far' scoring, since there should be fewer same values for 'far' instances)
                #                 diff_miss += 1 # Feature score is increased when we observe the same feature value between 'far' instances with different class values.
            # # *** NEW: 1st tweak attempt experimenting on compute_score(), near neighbors and binary output
            # # NEW: moving near check outside of NN loop
            # if near: # SCORING FOR NEAR INSTANCES
            #     # NEW: bringing ftype check outside of NN loop
            #     if ftype == 'continuous':
            #         # NEW: avoiding NN[i] lookup throughout this for loop
            #         for nn in NN: 
            #             if nan_entries[nn][feature]:  # skip any NN with a missing value for this feature.
            #                 continue

            #             xNNifeature = X[nn][feature]

            #             # if near:  # SCORING FOR NEAR INSTANCES
            #             if y_inst == y[nn]:   # HIT
            #                 count_hit += 1
            #                 # diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff #Normalize absolute value of feature value difference by max-min value range for feature (so score update lies between 0 and 1)
            #                 diff_hit -= ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)
            #             else:  # MISS
            #                 count_miss += 1
            #                 #diff_miss += abs(xinstfeature - xNNifeature) / mmdiff
            #                 diff_miss += ramp_function(data_type, attr, fname,
            #                                                 xinstfeature, xNNifeature)
            #     else: # feature is not continuous
            #         # NEW: avoiding NN[i] lookup throughout this for loop
            #         for nn in NN: 
            #             if nan_entries[nn][feature]:  # skip any NN with a missing value for this feature.
            #                 continue

            #             xNNifeature = X[nn][feature]

            #             # if near:  # SCORING FOR NEAR INSTANCES
            #             if y_inst == y[nn]:   # HIT
            #                 count_hit += 1
            #                 if xinstfeature != xNNifeature:  # A difference in feature value is observed
            #                     # Feature score is reduced when we observe feature difference between 'near' instances with the same class.
            #                     diff_hit -= 1
            #             else:  # MISS
            #                 count_miss += 1
            #                 if xinstfeature != xNNifeature:  # A difference in feature value is observed
            #                     # Feature score is increase when we observe feature difference between 'near' instances with different class values.
            #                     diff_miss += 1

            # # *** END NEW
            # *** 3rd Tweak attempt: numpy masking
            if near:
                # Step 1: Filter neighbors with non-missing feature values
                valid = ~nan_entries[NN, feature]
                nn_valid = NN[valid]
                weights_valid = weights_inst[valid]

                if nn_valid.size == 0:
                    count_hit = 0
                    count_miss = 0
                    diff_hit = 0.0
                    diff_miss = 0.0
                else:
                    # Step 2: Gather neighbor feature values & labels
                    x_nn = X[nn_valid, feature]
                    y_nn = y[nn_valid]

                    # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                    hits = (y_nn == y_inst)
                    misses = ~hits
                    weights_hits = weights_valid[hits]
                    weights_misses = weights_valid[misses]

                    count_hit = hits.sum()
                    count_miss = misses.sum()

                    # Step 4: Score updates
                    if ftype == 'continuous':
                        # vectorized ramp function
                        # ramp_vec must accept arrays
                        # diff_hit -= ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[hits]).sum()
                        # diff_miss += ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[misses]).sum()

                        # hit_diffs = ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[hits])
                        # hit_diffs = weights_hits * hit_diffs # weight the contribution of each instance by its calculated weight
                        # diff_hit -= hit_diffs.sum()
                        diff_hit -= (weights_hits * ramp_vec(data_type, attr, fname,
                                            xinstfeature, x_nn[hits])).sum()

                        # miss_diffs = ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[misses])
                        # miss_diffs = weights_misses * miss_diffs
                        # diff_miss += miss_diffs.sum()
                        diff_miss += (weights_misses * ramp_vec(data_type, attr, fname,
                                            xinstfeature, x_nn[misses])).sum()
                    else:
                        # discrete feature
                        # diff_hit -= np.sum(x_nn[hits] != xinstfeature)
                        # diff_miss += np.sum(x_nn[misses] != xinstfeature)
                        diff_hit -= np.sum(weights_hits * (x_nn[hits] != xinstfeature))
                        diff_miss += np.sum(weights_misses * (x_nn[misses] != xinstfeature))
            else: # Far scoring
                # Step 1: Filter neighbors with non-missing feature values
                valid = ~nan_entries[NN, feature]
                nn_valid = NN[valid]
                weights_valid = weights_inst[valid]

                if nn_valid.size == 0:
                    count_hit = 0
                    count_miss = 0
                    diff_hit = 0.0
                    diff_miss = 0.0
                else:
                    # Step 2: Gather neighbor feature values & labels
                    x_nn = X[nn_valid, feature]
                    y_nn = y[nn_valid]

                    # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                    hits = (y_nn == y_inst)
                    misses = ~hits
                    weights_hits = weights_valid[hits]
                    weights_misses = weights_valid[misses]

                    count_hit = hits.sum()
                    count_miss = misses.sum()

                    # Step 4: Score updates
                    if ftype == 'continuous':
                        # vectorized ramp function
                        # ramp_vec must accept arrays
                        # for each hit or miss, -/+ 1 is added, so that's why count_hit/count_miss is subtracted
                        # diff_hit -= (count_hit - ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[hits]).sum())
                        diff_hit -= (count_hit - (weights_hits * ramp_vec(data_type, attr, fname,
                                            xinstfeature, x_nn[hits])).sum())
                        # diff_miss += (count_miss - ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[misses]).sum())
                        diff_miss += (count_miss - (weights_misses * ramp_vec(data_type, attr, fname,
                                            xinstfeature, x_nn[misses])).sum())
                    else:
                        # discrete feature
                        # diff_hit -= np.sum(x_nn[hits] == xinstfeature)
                        diff_hit -= np.sum(weights_hits * (x_nn[hits] == xinstfeature))
                        # diff_miss += np.sum(x_nn[misses] == xinstfeature)
                        diff_miss += np.sum(weights_misses * (x_nn[misses] == xinstfeature))

            """ Score Normalizations:
            *'n' normalization dividing by the number of training instances (this helps ensure that all final scores end up in the -1 to 1 range
            *'k','h','m' normalization dividing by the respective number of hits and misses in NN (after ignoring missing values), also helps account for class imbalance within nearest neighbor radius)"""
            if count_hit == 0.0 or count_miss == 0.0:  # Special case, avoid division error
                if count_hit == 0.0 and count_miss == 0.0:
                    return 0.0
                elif count_hit == 0.0:
                    diff = (diff_miss / count_miss) / datalen
                else:  # count_miss == 0.0
                    diff = (diff_hit / count_hit) / datalen
            else:  # Normal diff normalization
                diff = ((diff_hit / count_hit) + (diff_miss / count_miss)) / datalen

        #--------------------------------------------------------------------------
        elif ctype == 'multiclass':
            # class_store = dict() #only 'miss' classes will be stored
            # #missClassPSum = 0

            # for each in mcmap:
            #     if(each != y[inst]):  # Identify miss classes for current target instance.
            #         class_store[each] = [0, 0]
            #         #missClassPSum += mcmap[each]

            # for i in range(len(NN)):
            #     if nan_entries[NN[i]][feature]:  # skip any NN with a missing value for this feature.
            #         continue

            #     xNNifeature = X[NN[i]][feature]

            #     if near:  # SCORING FOR NEAR INSTANCES
            #         if(y[inst] == y[NN[i]]):  # HIT
            #             count_hit += 1
            #             if ftype == 'continuous':
            #                 #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff
            #                 diff_hit -= ramp_function(data_type, attr, fname, xinstfeature, xNNifeature) 
            #             else: #discrete feature
            #                 if xinstfeature != xNNifeature:
            #                     # Feature score is reduced when we observe feature difference between 'near' instances with the same class.
            #                     diff_hit -= 1
            #         else:  # MISS
            #             for missClass in class_store:
            #                 if(y[NN[i]] == missClass):  # Identify which miss class is present
            #                     class_store[missClass][0] += 1
            #                     if ftype == 'continuous':
            #                         #class_store[missClass][1] += abs(xinstfeature - xNNifeature) / mmdiff
            #                         class_store[missClass][1] += ramp_function(
            #                             data_type, attr, fname, xinstfeature, xNNifeature)
            #                     else:  # discrete feature
            #                         if xinstfeature != xNNifeature:
            #                             # Feature score is increase when we observe feature difference between 'near' instances with different class values.
            #                             class_store[missClass][1] += 1

            #     else:  # SCORING FOR FAR INSTANCES (ONLY USED BY MULTISURF* BASED ON HOW CODED)
            #         if(y[inst] == y[NN[i]]):  # HIT
            #             count_hit += 1
            #             if ftype == 'continuous':
            #                 #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff  #Hits differently add continuous value differences rather than subtract them 
            #                 diff_hit -= (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
            #             else: #discrete features
            #                 if xinstfeature == xNNifeature:
            #                     # Feature score is reduced when we observe the same feature value between 'far' instances with the same class.
            #                     diff_hit -= 1
            #         else:  # MISS
            #             for missClass in class_store:
            #                 if(y[NN[i]] == missClass):
            #                     class_store[missClass][0] += 1
            #                     if ftype == 'continuous':
            #                         #class_store[missClass][1] += abs(xinstfeature - xNNifeature) / mmdiff
            #                         class_store[missClass][1] += (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
            #                     else: #discrete feature
            #                         if xinstfeature == xNNifeature:
            #                             # Feature score is increased when we observe the same feature value between 'far' instances with different class values.
            #                             class_store[missClass][1] += 1
            # *** TWEAKED VERSION OF CODE USING bulk numpy/vector operations instead of for loops
            # class_store = dict() #only 'miss' classes will be stored
            # #missClassPSum = 0

            # for each in mcmap:
            #     if(each != y[inst]):  # Identify miss classes for current target instance.
            #         class_store[each] = [0, 0]
            #         #missClassPSum += mcmap[each]
            #only 'miss' classes will be stored
            class_store = {
                cls: [0, 0]
                for cls in mcmap
                if cls != y[inst]
            }

            if near:
                # Step 1: Filter neighbors with non-missing feature values
                valid = ~nan_entries[NN, feature]
                nn_valid = NN[valid]
                weights_valid = weights_inst[valid]

                if nn_valid.size == 0:
                    # No need to set count_miss and diff_miss for each miss class to 0 because they are already initialized to 0 in class_store
                    count_hit = 0
                    diff_hit = 0.0
                else:
                    # Step 2: Gather neighbor feature values & labels
                    x_nn = X[nn_valid, feature]
                    y_nn = y[nn_valid]

                    # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                    hits = (y_nn == y_inst)
                    count_hit = hits.sum()
                    # # For multiclass, need to keep track of count of each miss class
                    # for missClass in class_store:
                    #     misses = (y_nn == missClass)
                    #     count_miss = misses.sum()
                    #     class_store[missClass][0] = count_miss
                    misses = ~hits
                    weights_hits = weights_valid[hits]
                    weights_misses = weights_valid[misses]

                    # Step 4: Score updates
                    if ftype == 'continuous':
                        # vectorized ramp function
                        # ramp_vec must accept arrays
                        # diff_hit -= ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[hits]).sum()
                        diff_hit -= (weights_hits * ramp_vec(data_type, attr, fname,
                                            xinstfeature, x_nn[hits])).sum()
                        # # For multiclass, need to keep track of count and diff for each miss class
                        # for missClass in class_store:
                        #     misses = (y_nn == missClass)
                        #     count_miss = misses.sum()
                        #     class_store[missClass][0] = count_miss

                        #     diff_miss = 0.0
                        #     diff_miss += ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[misses]).sum()
                        #     class_store[missClass][1] = diff_miss
                        # miss_diffs = ramp_vec(data_type, attr, fname, xinstfeature, x_nn[misses])
                        miss_diffs = weights_misses * ramp_vec(data_type, attr, fname, xinstfeature, x_nn[misses])
                        classes, inv = np.unique(y_nn[misses], return_inverse=True)

                        counts = np.bincount(inv)
                        diff_sums = np.bincount(inv, weights=miss_diffs)
                        for cls, cnt, diff_sum in zip(classes, counts, diff_sums):
                            class_store[cls][0] = cnt
                            class_store[cls][1] = diff_sum

                    else:
                        # discrete feature
                        # diff_hit -= np.sum(x_nn[hits] != xinstfeature)
                        diff_hit -= np.sum(weights_hits * (x_nn[hits] != xinstfeature))
                        # # For multiclass, need to keep track of count and diff for each miss class
                        # for missClass in class_store:
                        #     misses = (y_nn == missClass)
                        #     count_miss = misses.sum()
                        #     class_store[missClass][0] = count_miss

                        #     diff_miss = 0.0
                        #     diff_miss += np.sum(x_nn[misses] != xinstfeature)
                        #     class_store[missClass][1] = diff_miss
                        # miss_diffs = (x_nn[misses] != xinstfeature).astype(float)
                        miss_diffs = weights_misses * (x_nn[misses] != xinstfeature).astype(float)
                        classes, inv = np.unique(y_nn[misses], return_inverse=True)

                        counts = np.bincount(inv)
                        diff_sums = np.bincount(inv, weights=miss_diffs)
                        for cls, cnt, diff_sum in zip(classes, counts, diff_sums):
                            class_store[cls][0] = cnt
                            class_store[cls][1] = diff_sum

            else: # Far scoring
                # Step 1: Filter neighbors with non-missing feature values
                valid = ~nan_entries[NN, feature]
                nn_valid = NN[valid]
                weights_valid = weights_inst[valid]

                if nn_valid.size == 0:
                    # No need to set count_miss and diff_miss for each miss class to 0 because they are already initialized to 0 in class_store
                    count_hit = 0
                    diff_hit = 0.0
                else:
                    # Step 2: Gather neighbor feature values & labels
                    x_nn = X[nn_valid, feature]
                    y_nn = y[nn_valid]

                    # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                    hits = (y_nn == y_inst)
                    count_hit = hits.sum()
                    misses = ~hits
                    weights_hits = weights_valid[hits]
                    weights_misses = weights_valid[misses]

                    # Step 4: Score updates
                    if ftype == 'continuous':
                        # vectorized ramp function
                        # ramp_vec must accept arrays
                        # diff_hit -= (count_hit - ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[hits]).sum())
                        diff_hit -= (count_hit - (weights_hits * ramp_vec(data_type, attr, fname,
                                            xinstfeature, x_nn[hits])).sum())
                        # # For multiclass, need to keep track of count and diff for each miss class
                        # for missClass in class_store:
                        #     misses = (y_nn == missClass)
                        #     count_miss = misses.sum()
                        #     class_store[missClass][0] = count_miss

                        #     diff_miss = 0.0
                        #     diff_miss += (1 - ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[misses]).sum())
                        #     class_store[missClass][1] = diff_miss
                        # miss_diffs = ramp_vec(data_type, attr, fname, xinstfeature, x_nn[misses])
                        miss_diffs = weights_misses * ramp_vec(data_type, attr, fname, xinstfeature, x_nn[misses])
                        classes, inv = np.unique(y_nn[misses], return_inverse=True)

                        counts = np.bincount(inv)
                        diff_sums = np.bincount(inv, weights=miss_diffs)
                        for cls, cnt, diff_sum in zip(classes, counts, diff_sums):
                            class_store[cls][0] = cnt
                            class_store[cls][1] = (cnt - diff_sum)
                    else:
                        # discrete feature
                        # diff_hit -= np.sum(x_nn[hits] == xinstfeature)
                        diff_hit -= np.sum(weights_hits * (x_nn[hits] == xinstfeature)) # create boolean array, multiply elementwise by weight_hits, then sum up the hits' contributions
                        # # For multiclass, need to keep track of count and diff for each miss class
                        # for missClass in class_store:
                        #     misses = (y_nn == missClass)
                        #     count_miss = misses.sum()
                        #     class_store[missClass][0] = count_miss

                        #     diff_miss = 0.0
                        #     diff_miss += np.sum(x_nn[misses] == xinstfeature)
                        #     class_store[missClass][1] = diff_miss
                        # miss_diffs = (x_nn[misses] == xinstfeature).astype(float)
                        miss_diffs = weights_misses * (x_nn[misses] == xinstfeature).astype(float)
                        classes, inv = np.unique(y_nn[misses], return_inverse=True)

                        counts = np.bincount(inv)
                        diff_sums = np.bincount(inv, weights=miss_diffs)
                        for cls, cnt, diff_sum in zip(classes, counts, diff_sums):
                            class_store[cls][0] = cnt
                            class_store[cls][1] = diff_sum

            """ Score Normalizations:
            *'n' normalization dividing by the number of training instances (this helps ensure that all final scores end up in the -1 to 1 range
            *'k','h','m' normalization dividing by the respective number of hits and misses in NN (after ignoring missing values), also helps account for class imbalance within nearest neighbor radius)
            * multiclass normalization - accounts for scoring by multiple miss class, so miss scores don't have too much weight in contrast with hit scoring. If a given miss class isn't included in NN
            then this normalization will account for that possibility. """
            # Miss component
            for each in class_store:
                count_miss += class_store[each][0]

            if count_hit == 0.0 and count_miss == 0.0:
                return 0.0
            else:
                if count_miss == 0:
                    pass
                else: #Normal diff normalization
                    for each in class_store: #multiclass normalization
                        diff += class_store[each][1] * (class_store[each][0] / count_miss) * len(class_store)# Contribution of given miss class weighted by it's observed frequency within NN set.
                    diff = diff / count_miss #'m' normalization
                
                #Hit component: with 'h' normalization
                if count_hit == 0:
                    pass
                else:
                    diff += (diff_hit / count_hit)

            diff = diff / datalen  # 'n' normalization

        #--------------------------------------------------------------------------
        else:  # CONTINUOUS endpoint
            # same_class_bound = labels_std

            # for i in range(len(NN)):
            #     if nan_entries[NN[i]][feature]:  # skip any NN with a missing value for this feature.
            #         continue

            #     xNNifeature = X[NN[i]][feature]

            #     if near:  # SCORING FOR NEAR INSTANCES
            #         if abs(y[inst] - y[NN[i]]) < same_class_bound:  # HIT approximation
            #             count_hit += 1
            #             if ftype == 'continuous':
            #                 #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff
            #                 diff_hit -= ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)
            #             else:  # discrete feature
            #                 if xinstfeature != xNNifeature:
            #                     # Feature score is reduced when we observe feature difference between 'near' instances with the same 'class'.
            #                     diff_hit -= 1
            #         else:  # MISS approximation
            #             count_miss += 1
            #             if ftype == 'continuous':
            #                 #diff_miss += abs(xinstfeature - xNNifeature) / mmdiff
            #                 diff_miss += ramp_function(data_type, attr, fname,
            #                                            xinstfeature, xNNifeature)
            #             else:  # discrete feature
            #                 if xinstfeature != xNNifeature:
            #                     # Feature score is increase when we observe feature difference between 'near' instances with different class value.
            #                     diff_miss += 1

            #     else:  # SCORING FOR FAR INSTANCES (ONLY USED BY MULTISURF* BASED ON HOW CODED)
            #         if abs(y[inst] - y[NN[i]]) < same_class_bound:  # HIT approximation
            #             count_hit += 1
            #             if ftype == 'continuous':
            #                 #diff_hit += abs(xinstfeature - xNNifeature) / mmdiff
            #                 diff_hit -= (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
            #             else: #discrete feature
            #                 if xinstfeature == xNNifeature:
            #                     # Feature score is reduced when we observe the same feature value between 'far' instances with the same class.
            #                     diff_hit -= 1
            #         else:  # MISS approximation
            #             count_miss += 1
            #             if ftype == 'continuous':
            #                 #diff_miss -= abs(xinstfeature - xNNifeature) / mmdiff
            #                 diff_miss += (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
            #             else: #discrete feature
            #                 if xinstfeature == xNNifeature:
            #                     # Feature score is increased when we observe the same feature value between 'far' instances with different class values.
            #                     diff_miss += 1
            # *** 3rd Tweak attempt: numpy masking (continuous endpoint version)
            same_class_bound = labels_std

            if near:
                # Step 1: Filter neighbors with non-missing feature values
                valid = ~nan_entries[NN, feature]
                nn_valid = NN[valid]
                weights_valid = weights_inst[valid]

                if nn_valid.size == 0:
                    count_hit = 0
                    count_miss = 0
                    diff_hit = 0.0
                    diff_miss = 0.0
                else:
                    # Step 2: Gather neighbor feature values & labels
                    x_nn = X[nn_valid, feature]
                    y_nn = y[nn_valid]

                    # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                    # hits = (y_nn == y_inst)
                    hits = (np.abs(y_inst - y_nn) < same_class_bound)
                    misses = ~hits
                    weights_hits = weights_valid[hits]
                    weights_misses = weights_valid[misses]

                    count_hit = hits.sum()
                    count_miss = misses.sum()

                    # Step 4: Score updates
                    if ftype == 'continuous':
                        # vectorized ramp function
                        # ramp_vec must accept arrays
                        # diff_hit -= ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[hits]).sum()
                        diff_hit -= (weights_hits * ramp_vec(data_type, attr, fname,
                                            xinstfeature, x_nn[hits])).sum()
                        # diff_miss += ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[misses]).sum()
                        diff_miss += (weights_misses * ramp_vec(data_type, attr, fname,
                                            xinstfeature, x_nn[misses])).sum()
                    else:
                        # discrete feature
                        # diff_hit -= np.sum(x_nn[hits] != xinstfeature)
                        diff_hit -= np.sum(weights_hits * (x_nn[hits] != xinstfeature))
                        # diff_miss += np.sum(x_nn[misses] != xinstfeature)
                        diff_miss += np.sum(weights_misses * (x_nn[misses] != xinstfeature))
            else: # Far scoring
                # Step 1: Filter neighbors with non-missing feature values
                valid = ~nan_entries[NN, feature]
                nn_valid = NN[valid]
                weights_valid = weights_inst[valid]

                if nn_valid.size == 0:
                    count_hit = 0
                    count_miss = 0
                    diff_hit = 0.0
                    diff_miss = 0.0
                else:
                    # Step 2: Gather neighbor feature values & labels
                    x_nn = X[nn_valid, feature]
                    y_nn = y[nn_valid]

                    # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                    # hits = (y_nn == y_inst)
                    hits = (np.abs(y_inst - y_nn) < same_class_bound)
                    misses = ~hits
                    weights_hits = weights_valid[hits]
                    weights_misses = weights_valid[misses]

                    count_hit = hits.sum()
                    count_miss = misses.sum()

                    # Step 4: Score updates
                    if ftype == 'continuous':
                        # vectorized ramp function
                        # ramp_vec must accept arrays
                        # for each hit or miss, -/+ 1 is added, so that's why count_hit/count_miss is subtracted
                        # diff_hit -= (count_hit - ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[hits]).sum())
                        diff_hit -= (count_hit - (weights_hits * ramp_vec(data_type, attr, fname,
                                            xinstfeature, x_nn[hits])).sum())
                        # diff_miss += (count_miss - ramp_vec(data_type, attr, fname,
                        #                     xinstfeature, x_nn[misses]).sum())
                        diff_miss += (count_miss - (weights_misses * ramp_vec(data_type, attr, fname,
                                            xinstfeature, x_nn[misses])).sum())
                    else:
                        # discrete feature
                        # diff_hit -= np.sum(x_nn[hits] == xinstfeature)
                        diff_hit -= np.sum(weights_hits * (x_nn[hits] == xinstfeature))
                        # diff_miss += np.sum(x_nn[misses] == xinstfeature)
                        diff_miss += np.sum(weights_misses * (x_nn[misses] == xinstfeature))

            """ Score Normalizations:
            *'n' normalization dividing by the number of training instances (this helps ensure that all final scores end up in the -1 to 1 range
            *'k','h','m' normalization dividing by the respective number of hits and misses in NN (after ignoring missing values), also helps account for class imbalance within nearest neighbor radius)"""

            if count_hit == 0.0 or count_miss == 0.0:  # Special case, avoid division error
                if count_hit == 0.0 and count_miss == 0.0:
                    return 0.0
                elif count_hit == 0.0:
                    diff = (diff_miss / count_miss) / datalen
                else:  # count_miss == 0.0
                    diff = (diff_hit / count_hit) / datalen
            else:  # Normal diff normalization
                diff = ((diff_hit / count_hit) + (diff_miss / count_miss)) / datalen

        return diff
    
    @staticmethod
    def BaseSWRF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, headers, class_type, X, y, labels_std, data_type, weights_near, weights=None):
        scores = np.zeros(num_attributes)
        if isinstance(weights, np.ndarray):
            for feature_num in range(num_attributes):
                if len(NN_near) > 0:
                    scores[feature_num] += weights[feature_num] * BaseSWRF.compute_score(attr, mcmap, NN_near, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, weights_near)
        else:
            for feature_num in range(num_attributes):
                if len(NN_near) > 0:
                    scores[feature_num] += BaseSWRF.compute_score(attr, mcmap, NN_near, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, weights_near)

        return scores
    
    @staticmethod
    def BaseSWRFstar_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, NN_far, headers, class_type, X, y, labels_std, data_type, weights_near, weights_far, weights=None):
        scores = np.zeros(num_attributes)

        if isinstance(weights, np.ndarray):
            for feature_num in range(num_attributes):
                if len(NN_near) > 0:
                    scores[feature_num] += weights[feature_num] * BaseSWRF.compute_score(attr, mcmap, NN_near, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, weights_near)
                # Note that we add this term because we used the far scoring above by setting 'near' to False.  This is in line with original MultiSURF* paper.
                if len(NN_far) > 0:
                    scores[feature_num] += weights[feature_num] * BaseSWRF.compute_score(attr, mcmap, NN_far, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, weights_far, near=False)
        else:
            for feature_num in range(num_attributes):
                if len(NN_near) > 0:
                    scores[feature_num] += BaseSWRF.compute_score(attr, mcmap, NN_near, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, weights_near)
                # Note that we add this term because we used the far scoring above by setting 'near' to False.  This is in line with original MultiSURF* paper.
                if len(NN_far) > 0:
                    scores[feature_num] += BaseSWRF.compute_score(attr, mcmap, NN_far, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, weights_far, near=False)

        return scores

    def _run_algorithm(self):
        self.distance_weight_log = []  # Reset log before run
        self.instance_dist_stats = []
        # NEW: also reset this:
        self.std_weight_log = []

        # start_time = time.time()
        nan_entries = np.isnan(self._X)
        # print("Time taken to create nan_mask:", time.time() - start_time, "seconds\n")
        dists_flat = np.concatenate([np.array(row) for row in self._distance_array])
        # print("Time taken to create dists_flat:", time.time() - start_time, "seconds\n")
        mean_dist = dists_flat.mean()
        # print("Time taken to create mean_dist:", time.time() - start_time, "seconds\n")
        # print("Mean dist:", mean_dist, "\n")
        std_dist = dists_flat.std()
        # print("Time taken to create std_dist:", time.time() - start_time, "seconds\n")
        # print("STD of the distances:", std_dist, "\n")

        # NN_near_list, weights_near_list, NN_far_list, weights_far_list = [self._find_neighbors(datalen, mean_dist, std_dist) for datalen in range(self._datalen)]
        NN_near_list, weights_near_list, NN_far_list, weights_far_list = zip(*[self._find_neighbors(datalen, mean_dist, std_dist)
                                                                               for datalen in range(self._datalen)])

        if self.ignore_far: # only near neighbors
            if isinstance(self._weights, np.ndarray) and self.weight_final_scores:
                scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                    self.BaseSWRF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                            NN_near, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type, weights_near, self._weights)
                                                            for instance_num, NN_near, weights_near in zip(range(self._datalen), NN_near_list, weights_near_list)), axis=0)
            else:
                scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                    self.BaseSWRF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                            NN_near, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type, weights_near)
                                                            for instance_num, NN_near, weights_near in zip(range(self._datalen), NN_near_list, weights_near_list)), axis=0)
        else: # star algorithm that uses far neighbors
            if isinstance(self._weights, np.ndarray) and self.weight_final_scores:
                scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                    self.BaseSWRFstar_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                                NN_near, NN_far, self._headers, self._class_type, self._X, self._y,
                                                self._labels_std, self.data_type, weights_near, weights_far, self._weights)
                                                            for instance_num, NN_near, NN_far, weights_near, weights_far in
                                                            zip(range(self._datalen), NN_near_list, NN_far_list, weights_near_list, weights_far_list)), axis=0)
            else:
                scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                    self.BaseSWRFstar_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                                NN_near, NN_far, self._headers, self._class_type, self._X, self._y,
                                                self._labels_std, self.data_type, weights_near, weights_far)
                                                            for instance_num, NN_near, NN_far, weights_near, weights_far in
                                                            zip(range(self._datalen), NN_near_list, NN_far_list, weights_near_list, weights_far_list)), axis=0)
                
        return np.array(scores)
    
    def plot_distance_weight_map(self, save_fig=None, show_expected=True):
        """Visualize actual (distance, weight) pairs collected during Relief run."""
        if not self.distance_weight_log:
            print("No data logged yet. Run the algorithm first.")
            return

        distances, weights = zip(*self.distance_weight_log)
        # NEW: use self.std_weight_log for plotting
        distances_std, weights_std = zip(*self.std_weight_log)
        plt.figure(figsize=(10, 6))
        # plt.scatter(distances, weights, alpha=0.3, s=10, label='Observed')
        plt.scatter(distances_std, weights_std, alpha=0.3, s=10, label='Observed')

        if show_expected:
            if 'MultiSWRF' in self.name: 
                # Star variant → average per-instance mean/std
                x_vals = np.linspace(min(distances), max(distances), 500)
                means, stds, deadband = zip(*self.instance_dist_stats)
                mean_dist = np.mean(means)
                std_dist = np.mean(stds)
                dead_band = np.mean(deadband)
                # dead_band = std_dist / 4.0 if 'TBD' in self.name else 0
                near_threshold = mean_dist - dead_band
                far_threshold = mean_dist + dead_band

                # NEW: for plotting in terms of STD
                x_vals_std = (x_vals - mean_dist) / std_dist
            else:
                x_vals = np.linspace(min(distances), max(distances), 500)
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                dead_band = std_dist / 4.0 if 'MultiSWRF' in self.name else 0

                # NEW: for plotting in terms of STD
                x_vals_std = (x_vals - mean_dist) / std_dist

            if self.name in ('SWRF', 'SWRF*'):
                near_xvals = x_vals[x_vals < mean_dist]
                far_xvals  = x_vals[x_vals >= mean_dist]
                # y_vals = swrf_weight(x_vals, mean_dist, std_dist, dead_band)
                # y_vals = swrf_weight(x_vals, std_dist, mean_dist)
                y_near = swrf_weight(near_xvals, std_dist, mean_dist)
                y_far  = -1 * swrf_weight(far_xvals, std_dist, mean_dist)

                y_vals = np.concatenate([y_near, y_far])
            elif self.name in ('MultiSWRF', 'MultiSWRF*'):
                # y_vals = multiswrf_weight(x_vals, mean_dist, std_dist, dead_band)
                # y_vals = multiswrf_weight(x_vals, std_dist, mean_dist)
                near_xvals = x_vals[x_vals < mean_dist]
                far_xvals  = x_vals[x_vals >= mean_dist]

                y_near = multiswrf_weight(near_xvals, std_dist, mean_dist)
                y_far  = -1 * multiswrf_weight(far_xvals, std_dist, mean_dist)

                y_vals = np.concatenate([y_near, y_far])
            elif 'MultiSWRFDB' in self.name:
                near_xvals = x_vals[x_vals < near_threshold]
                zeroweight_xvals = x_vals[(x_vals >= near_threshold) & (x_vals <= far_threshold)]
                y_zero = np.zeros_like(zeroweight_xvals)
                far_xvals  = x_vals[x_vals > far_threshold]
                if 'linear' in self.name:
                    if '3SD' in self.name:
                        # y_vals = multiswrfdb_linear_3sd_weight(x_vals, mean_dist, std_dist, dead_band)
                        y_near = multiswrfdb_linear_3sd_weight(near_xvals, std_dist, near_threshold)
                        y_far = -1 * multiswrfdb_linear_3sd_weight(far_xvals, std_dist, far_threshold)
                        y_vals = np.concatenate([y_near, y_zero, y_far])
                    else:
                        # y_vals = multiswrfdb_linear_weight(x_vals, mean_dist, std_dist, dead_band)
                        y_near = multiswrfdb_linear_weight(near_xvals, std_dist, near_threshold)
                        y_far = -1 * multiswrfdb_linear_weight(far_xvals, std_dist, far_threshold)
                        y_vals = np.concatenate([y_near, y_zero, y_far])
                elif 'exponential' in self.name:
                    if '3SD' in self.name:
                        # y_vals = multiswrfdb_exponential_3sd_weight(x_vals, mean_dist, std_dist, dead_band)
                        y_near = multiswrfdb_exponential_3sd_weight(near_xvals, std_dist, near_threshold)
                        y_far = -1 * multiswrfdb_exponential_3sd_weight(far_xvals, std_dist, far_threshold)
                        y_vals = np.concatenate([y_near, y_zero, y_far])
                    else:
                        # y_vals = multiswrfdb_exponential_weight(x_vals, mean_dist, std_dist, dead_band)
                        y_near = multiswrfdb_exponential_weight(near_xvals, std_dist, near_threshold)
                        y_far = -1 * multiswrfdb_exponential_weight(far_xvals, std_dist, far_threshold)
                        y_vals = np.concatenate([y_near, y_zero, y_far])
                else:
                    # y_vals = multiswrfdb_weight(x_vals, mean_dist, std_dist, dead_band)
                    y_near = multiswrfdb_weight(near_xvals, std_dist, near_threshold)
                    y_far = -1 * multiswrfdb_weight(far_xvals, std_dist, far_threshold)
                    y_vals = np.concatenate([y_near, y_zero, y_far])
            else:
                y_vals = None

            # Apply ignore_far logic
            if self.ignore_far:
                for i, d in enumerate(x_vals):
                    if d > mean_dist:
                        y_vals[i] = 0.0

            if y_vals is not None:
                # plt.plot(x_vals, y_vals, label='Expected', linewidth=2, color='black')
                # NEW: use x_vals STD instead
                plt.plot(x_vals_std, y_vals, label='Expected', linewidth=2, color='black')

        plt.title(f'Distance-to-Weight Mapping: {self.name}')
        plt.xlabel('Distance from Target Instance')
        # plt.xlabel('Standard Deviations (Distance) from Target Instance')
        plt.ylabel('Scoring Weight')
        plt.grid(True)
        # NEW: grid lines different from x-tick labels:
        plt.gca().set_xticks(np.arange(-3, 4, 1), minor=False)
        plt.ylim(-1.1, 1.1)
        # NEW: xlim to set x-axis values between 0 and 1.0 for all graphs (consistent)
        # plt.xlim(0, 1.0)
        plt.xlim(-3.0, 3.0)
        # plt.xticks(np.linspace(0, 1.0, num=6))
        # plt.xticks([-3, -2, -1, 0, 1, 2, 3])
        plt.xticks(
            [-0.5, 0, 0.5],
            ['(μ - σ/2)', 'μ', '(μ + σ/2)']
        )
        # NEW: dotted lines for deadband zone boundaries (0.5 SD on either side of mean)
        plt.axvline(x=-0.5, color='red', linestyle='dotted')
        plt.axvline(x=0.5,  color='red', linestyle='dotted')
        plt.legend()
        if save_fig:
            plt.savefig(save_fig)
        else:
            plt.show()




class SWRFstar2(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('SWRF*', swrf_weight, ignore_far=False, **kwargs)


class SWRF(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('SWRF', swrf_weight, ignore_far=True, **kwargs)


class MultiSWRF(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRF', multiswrf_weight, ignore_far=True, **kwargs)


class MultiSWRFstar(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRF*', multiswrf_weight, ignore_far=False, **kwargs)


class MultiSWRFDB(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRFDB', multiswrfdb_weight, ignore_far=True, **kwargs)


class MultiSWRFDBstar(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRFDB*', multiswrfdb_weight, ignore_far=False, **kwargs)

class MultiSWRFDBlinear(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRFDB_linear', multiswrfdb_linear_weight, ignore_far=True, **kwargs)

class MultiSWRFDBlinearstar(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRFDB_linear*', multiswrfdb_linear_weight, ignore_far=False, **kwargs)

class MultiSWRFDBexponential(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRFDB_exponential', multiswrfdb_exponential_weight, ignore_far=True, **kwargs)

class MultiSWRFDBexponentialstar(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRFDB_exponential*', multiswrfdb_exponential_weight, ignore_far=False, **kwargs)

# 3 SD versions of MultiSWRFDB variants:
class MultiSWRFDBlinear3SD(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRFDB_linear_3SD', multiswrfdb_linear_3sd_weight, ignore_far=True, **kwargs)

class MultiSWRFDBlinear3SDstar(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRFDB_linear_3SD*', multiswrfdb_linear_3sd_weight, ignore_far=False, **kwargs)

class MultiSWRFDBexponential3SD(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRFDB_exponential_3SD', multiswrfdb_exponential_3sd_weight, ignore_far=True, **kwargs)

class MultiSWRFDBexponential3SDstar(BaseSWRF):
    def __init__(self, **kwargs):
        super().__init__('MultiSWRFDB_exponential_3SD*', multiswrfdb_exponential_3sd_weight, ignore_far=False, **kwargs)