from sklearn.base import BaseEstimator
import copy
import random
import numpy as np
from joblib import Parallel, delayed
import time
from itertools import combinations

class VLS(BaseEstimator):

    def __init__(self,relief_object,num_feature_subset=40,size_feature_subset=5,ensure_pair_coverage=False,
                 random_state = None,n_jobs=1):
        '''
        :param relief_object:           Must be an object that implements the standard sklearn fit function, and after fit, has attribute feature_importances_
                                        that can be accessed. Scores must be a 1D np.ndarray of length # of features. The fit function must also be able to
                                        take in an optional 1D np.ndarray 'weights' parameter of length num_features.
        :param num_feature_subset:      Number of feature subsets generated at random
        :param size_feature_subset:     Number of features in each subset. Cannot exceed number of features.
        :param ensure_pair_coverage:    Boolean indicating whether to generate feature subsets such that every pair of features 
                                        appears together in at least one subset. This is intended to ensure that all 2-way 
                                        interacting features are assigned together in at least one subset. If True, it overrides
                                        the value for num_feature_subset to generate the subsets needed.
        :param random_state:            random seed
        :param n_jobs:                  The number of cores to dedicate to completing operations with joblib. Assigning this 
                                        parameter to -1 will dedicate as many cores as are available on your system. We recommend 
                                        setting this parameter to -1 to speed up the algorithm as much as possible.
        '''

        if not self.check_is_int(num_feature_subset) or num_feature_subset <= 0:
            raise Exception('num_feature_subset must be a positive integer')

        if not self.check_is_int(size_feature_subset) or size_feature_subset <= 0:
            raise Exception('size_feature_subset must be a positive integer')
        
        if not self.check_is_bool(ensure_pair_coverage):
            raise Exception('ensure_pair_coverage must be a boolean value')

        if random_state != None and not self.check_is_int(random_state):
            raise Exception('random_state must be None or integer')

        self.relief_object = relief_object
        self.num_feature_subset = num_feature_subset
        self.size_feature_subset = size_feature_subset
        self.ensure_pair_coverage = ensure_pair_coverage
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.rank_absolute = self.relief_object.rank_absolute
        self.n_features_to_select = self.relief_object.n_features_to_select # for integration with TuRF transform method

    def fit(self, X, y, weights=None):
        """Scikit-learn required: Computes the feature importance scores from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features} Training instances to compute the feature importance scores from
        y: array-like {n_samples}             Training labels
        Returns
         -------
         self
        """
        #random_state
        if self.random_state != None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        #Make subsets with all the features
        num_features = X.shape[1]
        self.size_feature_subset = min(self.size_feature_subset,num_features)
        start_time = time.time()
        if self.ensure_pair_coverage:
            subsets = self.make_subsets_featuregroups(list(range(num_features)),self.size_feature_subset)
        else:
            subsets = self.make_subsets(list(range(num_features)),self.num_feature_subset,self.size_feature_subset)
        print("Time taken to generate subsets:", time.time() - start_time, " sec")

        # # UPDATE 5/11: making subsets in parallel
        # if self.num_feature_subset * self.size_feature_subset < num_features:
        #     raise Exception('num_feature_subset * size_feature_subset must be >= number of total features')

        # if self.size_feature_subset > num_features:
        #     raise Exception('size_feature_subset cannot be > number of total features')
        
        # possible_indices = list(range(num_features)) # indices of features able to be included in subsets
    
        # start_time = time.time()
        # subsets = list(Parallel(n_jobs=self.n_jobs)
        #                (delayed(self.make_subset)(possible_indices, self.size_feature_subset) for _ in range(self.num_feature_subset)))
        # print("Time taken to generate subsets:", time.time() - start_time, " sec")
        # start_time = time.time()
        # # END UPDATE 5/11

        # #Fit each subset
        # scores = []
        # for subset in subsets:
        #     new_X = self.custom_transform(X,subset)
        #     copy_relief_object = copy.deepcopy(self.relief_object)
        #     if not isinstance(weights,np.ndarray):
        #         copy_relief_object.fit(new_X,y)
        #     else:
        #         copy_relief_object.fit(new_X,y,weights=weights[subset])
        #     raw_score = copy_relief_object.feature_importances_
        #     score = np.empty(num_features)
        #     if self.rank_absolute:
        #         score.fill(0)
        #     else:
        #         score.fill(-np.inf)
        #     counter = 0
        #     for index in subset:
        #         score[index] = raw_score[counter]
        #         counter+=1
        #     scores.append(score)

        #     #DEBUGGING
        #     #print(score)

        # scores = np.array(scores)

        # #Merge results by selecting largest found weight for each feature
        # max_scores = []
        # for score in scores.T:
        #     if self.rank_absolute:
        #         max = np.max(np.absolute(score))
        #         if max in score:
        #             max_scores.append(max)
        #         else:
        #             max_scores.append(-max)
        #     else:
        #         max_scores.append(np.max(score))
        # max_scores = np.array(max_scores)
        
        # UPDATE 5/11: Fitting on subsets in parallel, extracting max score for each feature across all subsets
        # print("Time taken between subset generation and scoring of subsets:", time.time() - start_time, " sec")
        start_time = time.time()
        max_scores = np.max(Parallel(n_jobs=self.n_jobs)
                            (delayed(self.score_subset)(X, y, weights, num_features, subset) for subset in subsets), axis=0)
        print("Time taken to score subsets:", time.time() - start_time, " sec")
        # # END UPDATE 5/11

        #Save FI as feature_importances_
        self.feature_importances_ = max_scores

        if self.rank_absolute:
            self.top_features_ = np.argsort(np.absolute(self.feature_importances_))[::-1]
        else:
            self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        return self

    def custom_transform(self,X,indices_to_preserve):
        return X[:,indices_to_preserve]

    def make_subsets(self,possible_indices,num_feature_subset,size_feature_subset):
        if num_feature_subset * size_feature_subset < len(possible_indices):
            raise Exception('num_feature_subset * size_feature_subset must be >= number of total features')

        if size_feature_subset > len(possible_indices):
            raise Exception('size_feature_subset cannot be > number of total features')

        random.shuffle(possible_indices)
        remaining_indices = copy.deepcopy(possible_indices)

        subsets = []
        while True:
            subset = []
            while len(remaining_indices) > 0 and len(subset) < size_feature_subset:
                subset.append(remaining_indices.pop(0))
            subsets.append(subset)
            if len(remaining_indices) < size_feature_subset:
                break

        if len(remaining_indices) != 0:
            while len(remaining_indices) < size_feature_subset:
                index_bad = True
                while index_bad:
                    potential_index = random.choice(possible_indices)
                    if not (potential_index in remaining_indices):
                        remaining_indices.append(potential_index)
                        break
            subsets.append(remaining_indices)

        subsets_left = num_feature_subset - len(subsets)
        for i in range(subsets_left):
            subsets.append(random.sample(possible_indices,size_feature_subset))

        return subsets
    # def make_subset(self, possible_indices, size_feature_subset): # new version that makes one subset, called num_feature_subset times
    #     random.shuffle(possible_indices)

    #     subset = random.sample(possible_indices, size_feature_subset) # randomly generating a subset with size_feature_subset elements

    #     return subset

    def make_subsets_featuregroups(self,possible_indices,size_feature_subset):
        num_features = len(possible_indices)
        if size_feature_subset > num_features:
            raise Exception('size_feature_subset cannot be > number of total features')
        
        random.shuffle(possible_indices)
        remaining_indices = copy.deepcopy(possible_indices)

        num_feature_groups = int(np.ceil(num_features / (size_feature_subset / 2)))
        feature_groups = []

        # if size_feature_subset is divisible by 2:
        if size_feature_subset % 2 == 0:
            size_feature_group = size_feature_subset // 2 # integer division, result is an integer (not float)

            for _ in range(num_feature_groups): # for each feature group
                # create list for feature group
                feature_group = []

                # if there are enough unique features left to create a full feature group
                if len(remaining_indices) >= size_feature_group:
                    for _ in range(size_feature_group): # up until size_feature_group features are added to the feature group:
                        feature_group.append(remaining_indices.pop(0))

                    feature_groups.append(feature_group) # once the feature group is 100% complete, add it to the list of feature groups
                    
                # if there are not enough unique features left to create this final feature group, readjust feature groups
                # i.e. take a feature from other feature groups round-robin until remaining_indices catches up in size
                else:
                    group_idx = 0

                    while len(remaining_indices) < min(len(g) for g in feature_groups):
                        # Take one feature from the current group
                        feature = feature_groups[group_idx].pop()

                        # Add it to remaining_indices
                        remaining_indices.append(feature)

                        # Move to the next group (round-robin)
                        group_idx = (group_idx + 1) % len(feature_groups)
                    
                    feature_groups.append(remaining_indices) # add remaining_indices to feature_groups as the final feature group

        # if size_feature_subset is NOT divisible by 2
        else:
            for idx in range(num_feature_groups): # for each feature group
                # on even iterations use floor function, on odd iterations use ceiling
                if idx % 2 == 0:
                    size_feature_group = int(np.floor(size_feature_subset / 2))
                else:
                    size_feature_group = int(np.ceil(size_feature_subset / 2))

                # create list for feature group
                feature_group = []

                # if there are enough unique features left to create a full feature group
                if len(remaining_indices) >= size_feature_group:
                    for _ in range(size_feature_group): # up until size_feature_group features are added to the feature group:
                        feature_group.append(remaining_indices.pop(0))

                    feature_groups.append(feature_group) # once the feature group is 100% complete, add it to the list of feature groups
                    
                # if there are not enough unique features left to create this final feature group, readjust feature groups
                # i.e. take a feature from other feature groups round-robin until remaining_indices catches up in size
                else:
                    group_idx = 0

                    while len(remaining_indices) < min(len(g) for g in feature_groups):
                        # Take one feature from the current group
                        feature = feature_groups[group_idx].pop()

                        # Add it to remaining_indices
                        remaining_indices.append(feature)

                        # Move to the next group (round-robin)
                        group_idx = (group_idx + 1) % len(feature_groups)
                    
                    feature_groups.append(remaining_indices) # add remaining_indices to feature_groups as the final feature group

        # Now, produce all 2 feature group combinations to form the subsets
        subsets = []

        for group1, group2 in combinations(feature_groups, 2):
            subsets.append(group1 + group2)

        return subsets        


    def score_subset(self, X, y, weights, num_features, subset):
        new_X = self.custom_transform(X,subset)
        copy_relief_object = copy.deepcopy(self.relief_object)
        if not isinstance(weights,np.ndarray):
            copy_relief_object.fit(new_X,y)
        else:
            copy_relief_object.fit(new_X,y,weights=weights[subset])
        raw_score = copy_relief_object.feature_importances_
        score = np.empty(num_features)
        if self.rank_absolute:
            score.fill(0)
        else:
            score.fill(-np.inf)
        counter = 0
        for index in subset:
            score[index] = raw_score[counter]
            counter+=1
        
        return score

        

    def check_is_int(self, num):
        try:
            n = float(num)
            if num - int(num) == 0:
                return True
            else:
                return False
        except:
            return False

    def check_is_float(self, num):
        try:
            n = float(num)
            return True
        except:
            return False
    
    def check_is_bool(self, obj):
        return isinstance(obj, bool)

    def transform(self, X):
        if X.shape[1] < self.relief_object.n_features_to_select:
            raise ValueError('Number of features to select is larger than the number of features in the dataset.')

        return X[:, self.top_features_[:self.relief_object.n_features_to_select]]

    def fit_transform(self, X, y, weights=None):
        self.fit(X, y, weights)
        return self.transform(X)