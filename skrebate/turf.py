from sklearn.base import BaseEstimator
import copy
import numpy as np
from .vls import VLS

class TURF(BaseEstimator):
    def __init__(self, relief_object, n_iterations=10, num_scores_to_return=10000):
        '''
        :param relief_object:           Must be an object that implements the standard sklearn fit function, and after fit, has attributes feature_importances_
                                        and top_features_ that can be accessed. Scores must be a 1D np.ndarray of length # of features.
        :param n_iterations:            Number of iterations to remove features for. The number of features removed each iteration
                                        is determined by (a - num_scores_to_return) / (n_iterations), with a equal to the total number
                                        of features in the dataset.    
        :param num_scores_to_return:    Number of nonzero scores to return after training. Default = min(num_features, 10000)
        '''
        if not self.check_is_int(num_scores_to_return) or num_scores_to_return <= 0 or num_scores_to_return > 10000:
            raise Exception('num_scores_to_return must be a positive integer <= 10,000')

        if not self.check_is_int(n_iterations) or n_iterations <= 0:
            raise Exception('n_iterations must be a positive integer')

        self.relief_object = relief_object
        self.n_iterations = n_iterations
        self.num_scores_to_return = num_scores_to_return
        self.rank_absolute = self.relief_object.rank_absolute

    def fit(self, X, y):
        """Scikit-learn required: Computes the feature importance scores from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features} Training instances to compute the feature importance scores from
        y: array-like {n_samples}             Training labels
        Returns
         -------
         self
        """
        #Adjust num_scores_to_return
        num_features = X.shape[1]
        self.num_scores_to_return = min(self.num_scores_to_return,num_features)

        # Number of features to eliminate each iteration
        num_features_to_eliminate = np.floor((num_features - self.num_scores_to_return) / self.n_iterations)

        # if TuRF is using VLS as its inner algorithm, calculate proportion of total features included in each subset
        # ... to ensure this remains constant across the iterations
        if isinstance(self.relief_object, VLS):
            subset_proportion = self.relief_object.size_feature_subset / num_features

        #Find out how many features to use in each iteration
        current_features = list(range(num_features))
        eliminated_tiers = []

        if self.num_scores_to_return != num_features: # if num_scores_to_return == num_features, no removal of features and move onto final fit iteration
            for i in range(self.n_iterations):
                X_subset = X[:, current_features]
                relief = copy.deepcopy(self.relief_object)

                if isinstance(relief, VLS) and i != 0: # adjust the number of features in each VLS subset
                    relief.size_feature_subset = int(np.ceil(subset_proportion * len(current_features)))

                relief.fit(X_subset, y)

                importances = relief.feature_importances_
                ranks = np.argsort(importances)[::-1]
                if i == (self.n_iterations - 1): # if it's the last iteration, ensure exactly num_scores_to_return features are kept
                    keep_count = self.num_scores_to_return
                else:
                    keep_count = int(len(current_features) - num_features_to_eliminate)
                keep_indices = sorted(ranks[:keep_count])
                eliminate_indices = [j for j in range(len(current_features)) if j not in keep_indices]

                eliminated_tiers.append([current_features[j] for j in eliminate_indices])
                current_features = [current_features[j] for j in keep_indices]

        # Final round on the reduced set
        X_final = X[:, current_features]
        relief = copy.deepcopy(self.relief_object)

        if isinstance(relief, VLS): # adjust the number of features in each VLS subset
            relief.size_feature_subset = int(np.ceil(subset_proportion * len(current_features)))

        relief.fit(X_final, y)
        final_scores = relief.feature_importances_

        # Build full-length feature_importances_ with tier scores for eliminated features
        full_scores = np.zeros(num_features)
        full_scores[current_features] = final_scores

        # Tier score fallback
        min_score = min(final_scores)
        max_score = max(final_scores)
        score_range = max_score - min_score if max_score != min_score else 1.0
        tier_penalty = 0.01 * score_range

        for k, tier in enumerate(reversed(eliminated_tiers)):
            tier_score = min_score - tier_penalty * (k + 1)
            for idx in tier:
                full_scores[idx] = tier_score

        self.feature_importances_ = full_scores
        self.top_features_ = np.argsort(np.abs(full_scores) if self.rank_absolute else full_scores)[::-1]
        return self


    def check_is_int(self, num):
        try:
            return float(num).is_integer()
        except:
            return False

    def check_is_float(self, num):
        try:
            float(num)
            return True
        except:
            return False

    def transform(self, X):
        if X.shape[1] < self.relief_object.n_features_to_select:
            raise ValueError('Number of features to select is larger than the number of features in the dataset.')

        return X[:, self.top_features_[:self.relief_object.n_features_to_select]]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)