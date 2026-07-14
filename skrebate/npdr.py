# from sklearn.base import BaseEstimator
from .relieff import ReliefF
from .surf import SURF
from .surfstar import SURFstar
from .multisurf import MultiSURF
from .multisurfstar import MultiSURFstar
from .baseswrf import SWRF, SWRFstar, MultiSWRF, MultiSWRFstar, MultiSWRFDB, MultiSWRFDBstar
from .murelief import MuRelief
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import t, norm
from joblib import Parallel, delayed
import warnings
import time

class NPDR(ReliefF):

    """Feature selection algorithm designed for continuous outcome data (but also handles binary outcomes).
    Based on the NPDR algorithm as introduced in:
    T.T. Le, B.A. Dawkins, and B.A. McKinney
    Nearest-neighbor Projected-Distance Regression (NPDR) for detecting network interactions with adjustments for multiple tests and confounding
    """

    def __init__(self, n_features_to_select=10, label_type=None, relief_object=None, padj_method="fdr_bh", categorical_threshold=10, 
                 categorical_features=None, categorical_covariates=None, n_jobs=1):
        """Sets up NPDR to perform feature selection
        Parameters
        ----------
        n_features_to_select: int (default: 10)
            The number of top features (according to the feature importance score) to
            retain after feature selection is applied.
        label_type: str (default: None)
            The default value is None, in which case the function automatically infers the label type
            based on the number of unique labels/outcomes: 2 for 'binary', >2 for 'continuous' (multiclass is not supported).
            Alternatively, you can specify one of the following strings: 'binary' or 'continuous'.
            Used to determine whether logistic or linear regression should be run on the outcome variable.
        relief_object: object (default: None)
            A core Relief-based algorithm (RBA) that has a _find_neighbors function. Used to identify neighbor instances for each target.
        padj_method: str (default: "fdr_bh")
            Method used to adjust p-values for multiple testing.
        categorical_threshold: int (default: 10)
            Value used to determine if a feature is categorical/discrete or continuous.
            If the number of unique values in a feature is > categorical_threshold, then it is
            considered continuous, and categorical otherwise.
        categorical_features: list (default: None)
            List of index columns indicating features to be treated as categorical.
            If set to None, the features will be automatically classified based on the categorical_threshold.
        categorical_covariates: list (default: None)
            List of index columns indicating covariates to be treated as categorical.
            If set to None, the covariates will be automatically classified based on the categorical_threshold.  
        n_jobs: int (default: 1)
            The number of cores to dedicate to computing the scores with joblib.
            Assigning this parameter to -1 will dedicate as many cores as are available on your system.
            We recommend setting this parameter to -1 to speed up the algorithm as much as possible.
        """

        self.n_features_to_select = n_features_to_select
        self.label_type = label_type
        self.relief_object = (MultiSWRFDB() if relief_object is None else relief_object)
        self.padj_method = padj_method
        self.categorical_threshold = categorical_threshold
        self.categorical_features = categorical_features
        self.categorical_covariates = categorical_covariates
        self.n_jobs = n_jobs

        self._validate_params()

    def _validate_params(self):
        if not isinstance(self.n_features_to_select, int):
            raise TypeError("n_features_to_select must be an integer")
        if self.n_features_to_select < 1:
            raise ValueError("n_features_to_select must be a positive integer.")

        if self.label_type is not None and self.label_type not in ("binary", "continuous"):
            raise ValueError(
                "label_type must be either 'binary', 'continuous', or None for automatic selection"
            )

        allowed_types = (
            ReliefF,
            SURF,
            SURFstar,
            MultiSURF,
            MultiSURFstar,
            SWRF,
            SWRFstar,
            MultiSWRF,
            MultiSWRFstar,
            MultiSWRFDB,
            MultiSWRFDBstar,
            MuRelief,
        )
        if not isinstance(self.relief_object, allowed_types):
            raise TypeError(
                "relief_object must be a core RBA from the skrebate package"
            )
        if self.label_type == "continuous" and type(self.relief_object) in (ReliefF, MuRelief):
            raise ValueError(
                "If label_type is 'continuous', a radius-based relief_object must be used"
            )

        allowed_methods = (
            "bonferroni",
            "sidak",
            "holm-sidak",
            "holm",
            "fdr_bh",
            "fdr_by",
            "fdr_tsbh",
            "fdr_tsbky",
        )
        if self.padj_method not in allowed_methods:
            raise ValueError(
                f"padj_method must be one of {allowed_methods}, got '{self.padj_method}'"
            )
        
        if not isinstance(self.categorical_threshold, int):
            raise TypeError("categorical_threshold must be an integer")
        if self.categorical_threshold < 2:
            raise ValueError("categorical_threshold must be >= 2")
        
        if self.categorical_features is not None and not isinstance(self.categorical_features, list):
            raise TypeError("categorical_features must be a list of feature indices or None")
        
        if self.categorical_covariates is not None and not isinstance(self.categorical_covariates, list):
            raise TypeError("categorical_covariates must be a list of covariate indices or None")

        if not isinstance(self.n_jobs, int):
            raise TypeError("n_jobs must be an integer")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be a positive integer or -1")
        
    def fit(self, X, y, covariates=None):
        """Scikit-learn required: Computes the feature importance scores (standardized beta's) from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training outcomes
        covariates: None or array-like {n_samples, n_covariates}
            Covariates in per-attribute regression models. None or a matrix of covariate values.

        Returns
        -------
        Copy of the NPDR instance
        """

        self._X = X  # matrix of predictive variables ('independent variables')
        self._y = y  # vector of values for outcome variable ('dependent variable')
        self._covariates = covariates # matrix of covariates

        self._datalen = len(self._X)  # Number of training instances ('n')

        # Number of unique outcome values (used to determine label/outcome type if user doesn't specify)
        self._label_list = list(set(self._y))

        if len(self._label_list) == 1:
            raise ValueError('All labels are of the same class.')
        # if label_type is provided:
        if self.label_type is not None:
            if self.label_type == 'binary' and len(self._label_list) != 2:
                raise ValueError("Specified 'binary' label type, but the number of unique labels/outcomes is not 2.") 
            self._class_type = self.label_type
        # if label_type is NOT provided, auto-detect:
        else:
            if len(self._label_list) == 2:
                self._class_type = 'binary'
                print("Class type = binary")
            elif len(self._label_list) <= 10:
                self._class_type = 'continuous'
                warnings.warn(
                    "Detected between 3 and 10 unique outcome values. Multiclass data is not supported; encoding as continuous outcome.",
                    UserWarning,
                )
                if type(self.relief_object) in (ReliefF, MuRelief):
                    raise ValueError(
                        "Detected continuous-valued outcome, so a radius-based relief_object must be used"
                    )
                print("Class type = continuous")
            else:
                self._class_type = 'continuous'
                if type(self.relief_object) in (ReliefF, MuRelief):
                    raise ValueError(
                        "Detected continuous-valued outcome, so a radius-based relief_object must be used"
                    )
                print("Class type = continuous")

        self._num_attributes = len(self._X[0])  # Number of predictors in training data

        # Number of missing data values in predictor variable matrix.
        self._missing_data_count = np.isnan(self._X).sum()

        """Assign internal headers for the features (scikit-learn does not accept external headers from dataset):
        The pre_normalize() function relies on the headers being ordered, e.g., X01, X02, etc.
        If this is changed, then the sort in the pre_normalize() function needs to be adapted as well. """
        xlen = len(self._X[0])
        mxlen = len(str(xlen + 1))
        self._headers = ['X{}'.format(str(i).zfill(mxlen)) for i in range(1, xlen + 1)]

        # Determine data types for all features/attributes in training data (i.e. categorical or continuous)
        C = D = False
        # Examines each feature and applies categorical_threshold to determine variable type (or uses categorical_features passed in).
        self.attr = self._get_attribute_info()
        for key in self.attr.keys():
            if self.attr[key][0] == 'categorical':
                D = True
            if self.attr[key][0] == 'continuous':
                C = True

        # For downstream computational efficiency, determine if dataset is comprised of all categorical, all continuous, or a mix of categorical/continuous features.
        if C and D:
            self.data_type = 'mixed'
        elif D and not C:
            self.data_type = 'categorical'
        elif C and not D:
            self.data_type = 'continuous'
        else:
            raise ValueError('Invalid data type in data set.')
        
        # Compute the distance array between all data points ----------------------------------------------------------------
        # For downstream efficiency, separate features in dataset by type (i.e. categorical/continuous); categorical = didx (discrete)
        diffs, cidx, didx = self._dtype_array()
        cdiffs = diffs[cidx]  # max/min continuous value difference for continuous features.

        xc = self._X[:, cidx]  # Subset of continuous-valued feature data
        xd = self._X[:, didx]  # Subset of categorical-valued feature data

        self.distarray_has_nan = False
        """ For efficiency, the distance array is computed more efficiently for data with no missing values.
        This distance array will only be used to identify nearest neighbors. """
        if self._missing_data_count > 0:
            self._distance_array = self._distarray_missing(xc, xd, cdiffs)
            
            # if distance array has nan values, will use np.nanmean/np.nanstd downstream
            if np.isnan(self._distance_array).any():
                self.distarray_has_nan = True
        else:
            self._distance_array = self._distarray_no_missing(xc, xd)

        # needed since find_neighbors() is in context of self.relief_object
        self.relief_object._distance_array = self._distance_array
        self.relief_object._datalen = self._datalen
        self.relief_object._label_list = self._label_list
        self.relief_object._class_type = self._class_type
        self.relief_object.distarray_has_nan = self.distarray_has_nan
        
        # list to contain all neighbor pairs across all target instances
        global_neighborhood_pairs = [] 

        # Checking what type of instance self.relief_object is
        if type(self.relief_object) in (ReliefF, MuRelief, MultiSURF):
            neighbor_list = [self.relief_object._find_neighbors(datalen) for datalen in range(self._datalen)]

            # creating pairs for global neighborhood, each of the tuple form (target_idx, neighbor_idx)
            for target_idx, neighbors in enumerate(neighbor_list):
                for neighbor_idx in neighbors:
                    global_neighborhood_pairs.append((target_idx, neighbor_idx))

            print("Relief_object is ReliefF, Mu-Relief, or MultiSURF")

        elif type(self.relief_object) in (MultiSURFstar,):
            NNlist = [self.relief_object._find_neighbors(datalen) for datalen in range(self._datalen)]
            NN_near_list = [i[0] for i in NNlist]
            NN_far_list = [i[1] for i in NNlist]

            # creating pairs for global neighborhood, each of the tuple form (target_idx, neighbor_idx)
            for target_idx, neighbors in enumerate(NN_near_list):
                for neighbor_idx in neighbors:
                    global_neighborhood_pairs.append((target_idx, neighbor_idx))
            for target_idx, neighbors in enumerate(NN_far_list):
                for neighbor_idx in neighbors:
                    global_neighborhood_pairs.append((target_idx, neighbor_idx))

            print("Relief_object is MultiSURF*")
        
        elif type(self.relief_object) in (SURF,):
            dists_flat = np.concatenate([np.array(row) for row in self._distance_array])
            if self.distarray_has_nan:
                avg_dist = np.nanmean(dists_flat)
            else:
                avg_dist = dists_flat.mean()

            NNlist = [self.relief_object._find_neighbors(datalen, avg_dist) for datalen in range(self._datalen)]

            # creating pairs for global neighborhood, each of the tuple form (target_idx, neighbor_idx)
            for target_idx, neighbors in enumerate(NNlist):
                for neighbor_idx in neighbors:
                    global_neighborhood_pairs.append((target_idx, neighbor_idx))

            print("Relief_object is SURF")
        
        elif type(self.relief_object) in (SURFstar,):
            dists_flat = np.concatenate([np.array(row) for row in self._distance_array])
            if self.distarray_has_nan:
                avg_dist = np.nanmean(dists_flat)
            else:
                avg_dist = dists_flat.mean()

            NNlist = [self.relief_object._find_neighbors(datalen, avg_dist) for datalen in range(self._datalen)]
            NN_near_list = [i[0] for i in NNlist]
            NN_far_list = [i[1] for i in NNlist]

            # creating pairs for global neighborhood, each of the tuple form (target_idx, neighbor_idx)
            for target_idx, neighbors in enumerate(NN_near_list):
                for neighbor_idx in neighbors:
                    global_neighborhood_pairs.append((target_idx, neighbor_idx))
            for target_idx, neighbors in enumerate(NN_far_list):
                for neighbor_idx in neighbors:
                    global_neighborhood_pairs.append((target_idx, neighbor_idx))

            print("Relief_object is SURF*")

        elif type(self.relief_object) in (SWRF, MultiSWRF, MultiSWRFDB):
            dists_flat = np.concatenate([np.array(row) for row in self._distance_array])
            if self.distarray_has_nan:
                mean_dist = np.nanmean(dists_flat)
                std_dist = np.nanstd(dists_flat)
            else:
                mean_dist = dists_flat.mean()
                std_dist = dists_flat.std()

            NN_near_list, _, _, _ = zip(*[self.relief_object._find_neighbors(datalen, mean_dist, std_dist)
                                                                               for datalen in range(self._datalen)])
            
            # creating pairs for global neighborhood, each of the tuple form (target_idx, neighbor_idx)
            for target_idx, neighbors in enumerate(NN_near_list):
                for neighbor_idx in neighbors:
                    global_neighborhood_pairs.append((target_idx, neighbor_idx))

            print("Relief_object is SWRF, MultiSWRF, or MultiSWRFDB")
        
        elif type(self.relief_object) in (SWRFstar, MultiSWRFstar, MultiSWRFDBstar):
            dists_flat = np.concatenate([np.array(row) for row in self._distance_array])
            if self.distarray_has_nan:
                mean_dist = np.nanmean(dists_flat)
                std_dist = np.nanstd(dists_flat)
            else:
                mean_dist = dists_flat.mean()
                std_dist = dists_flat.std()

            NN_near_list, _, NN_far_list, _ = zip(*[self.relief_object._find_neighbors(datalen, mean_dist, std_dist)
                                                                               for datalen in range(self._datalen)])
            
            # creating pairs for global neighborhood, each of the tuple form (target_idx, neighbor_idx)
            for target_idx, neighbors in enumerate(NN_near_list):
                for neighbor_idx in neighbors:
                    global_neighborhood_pairs.append((target_idx, neighbor_idx))
            for target_idx, neighbors in enumerate(NN_far_list):
                for neighbor_idx in neighbors:
                    global_neighborhood_pairs.append((target_idx, neighbor_idx))

            print("Relief_object is SWRF*, MultiSWRF*, or MultiSWRFDB*")
        # global neighborhood computation completed
        # * global neighborhood will include both (a, b) and (b, a) pairs currently (just like core RBAs)
        # print("Global neighborhood pairs: \n", global_neighborhood_pairs)
        # print("Length of global_neighborhood_pairs:", len(global_neighborhood_pairs))

        # ensure y and covariates are np arrays
        y_arr = np.asarray(self._y)
        covariates_arr = np.asarray(self._covariates)

        # identifying whether covariates are continuous or categorical (if they exist)
        covariate_types = []
        if self._covariates is not None:
            if self.categorical_covariates is None: # categorical covariates are not specified, automatically identify
                for cov_idx in range(covariates_arr.shape[1]):
                    cov_values = covariates_arr[:, cov_idx]
                    cov_values = cov_values[~np.isnan(cov_values)]  # Exclude any missing values from consideration
                    unique_cov_values = np.unique(cov_values).size

                    if unique_cov_values <= self.categorical_threshold:
                        covariate_types.append("categorical")
                    else:
                        covariate_types.append("continuous")
            else: # categorical covariates are specified
                categorical_covariates = set(self.categorical_covariates) # list changed to set for faster lookup
                for cov_idx in range(covariates_arr.shape[1]):
                    if cov_idx in categorical_covariates:
                        covariate_types.append("categorical")
                    else:
                        covariate_types.append("continuous")

            print("Covariate types: \n", covariate_types)

        # # distance vector for y and covariates 
        # dist_y = []
        # dist_covariates = []

        # # * can parallelize this process
        # for i, j in global_neighborhood_pairs:
        #     # value target i has for y
        #     y_i = y_arr[i]
        #     # value neighbor j has for y
        #     y_j = y_arr[j]

        #     if self._class_type == "continuous":
        #         # difference between target and neighbor in outcome (continuous)
        #         diff_y = np.abs(y_i - y_j)
        #     else:
        #         # difference between target and neighbor in outcome (binary, different = 1 equal = 0)
        #         diff_y = 1 if y_i != y_j else 0

        #     # add distance in y between (target i and neighbor j) to y distance vector
        #     dist_y.append(diff_y)

        #     # if covariates is not None, add to distance vector/matrix for covariates
        #     if self._covariates is not None:
        #         # list to contain differences between current target and neighbor for all covariates (a "row" of differences, columns = # of covariates)
        #         cov_differences = []
        #         for cov_idx in range(covariates_arr.shape[1]):
        #             # values for current covariate 'cov' for target i and neighbor j
        #             cov_i = covariates_arr[i, cov_idx]
        #             cov_j = covariates_arr[j, cov_idx]

        #             if covariate_types[cov_idx] == "continuous":
        #                 # difference between target and neighbor in current covariate (continuous)
        #                 diff_cov = np.abs(cov_i - cov_j)
        #             else:
        #                 # difference between target and neighbor in current covariate (categorical, different = 1 equal = 0)
        #                 diff_cov = 1 if cov_i != cov_j else 0
                    
        #             # add difference for this covariate to the list of all covariate differences for current i and j
        #             cov_differences.append(diff_cov)

        #         dist_covariates.append(cov_differences) 
        
        # dist_y = np.array(dist_y)
        # dist_covariates = np.array(dist_covariates)

        # *** vectorized version of commented out code above (dist_y and dist_covariates computation)
        pairs = np.asarray(global_neighborhood_pairs)
        print("Pairs: \n", pairs)
        print("Pairs shape:", pairs.shape)

        # computing distance vector for y (outcome)
        if self._class_type == "continuous":
            # difference between targets and neighbors in outcome (continuous)
            dist_y = np.abs(y_arr[pairs[:, 0]] - y_arr[pairs[:, 1]])
        else:
            # difference between targets and neighbors in outcome (binary, different = 1 equal = 0)
            dist_y = (y_arr[pairs[:, 0]] != y_arr[pairs[:, 1]]).astype(int)

        # print("Dist_y: \n", dist_y)
        # print("Dist_y shape:", dist_y.shape)

        # computing distance matrix for covariates (if they exist)
        if self._covariates is not None:
            dist_covariates = np.empty((len(pairs), covariates_arr.shape[1]))

            for cov_idx, cov_type in enumerate(covariate_types):
                if cov_type == "continuous":
                    # difference between targets and neighbors in current covariate (continuous)
                    dist_covariates[:, cov_idx] = np.abs(
                        covariates_arr[pairs[:, 0], cov_idx] -
                        covariates_arr[pairs[:, 1], cov_idx]
                    )
                else:
                    # difference between targets and neighbors in current covariate (categorical, different = 1 equal = 0)
                    dist_covariates[:, cov_idx] = (
                        covariates_arr[pairs[:, 0], cov_idx] !=
                        covariates_arr[pairs[:, 1], cov_idx]
                    ).astype(int)

            print("Dist_covariates: \n", dist_covariates)
            print("Dist_covariates shape:", dist_covariates.shape)
        else:
            dist_covariates = None # set to None if self._covariates was not defined
        # *** end of vectorized version for dist_y and dist_covariates computation

        # will be used to identify feature type for each feature
        attr_types = [value[0] for value in self.attr.values()]
        print("Attr_types: \n", attr_types)
        # ensure X is np array
        X_arr = np.asarray(self._X)

        # looping through the columns of X (i.e. looping through the attributes to create per-attribute regression models)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._create_regression_model)(
                a,
                attr_types,
                X_arr,
                pairs,
                dist_covariates,
                dist_y
            )
            for a in range(self._X.shape[1])
        )
        # for a in range(self._X.shape[1]):

        betas = np.array([r[0] for r in results])
        z_betas = np.array([r[1] for r in results])
        pvalues = np.array([r[2] for r in results])

        # storing results as instance variables (z_betas are the final feature importance scores)
        self.beta_ = betas
        self.feature_importances_ = z_betas
        self.pvalues_ = pvalues

        pvalues_adj = multipletests(
            self.pvalues_,
            method=self.padj_method
        )[1]
        self.pvalues_adj_ = pvalues_adj # adjusted p-values

        # sorting feature indices based on feature importance score
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        return self

    def _create_regression_model(self, a, attr_types, X_arr, global_neighborhood_pairs, dist_covariates, dist_y):
        # # distance vector for feature a
        # dist_a = []
        # # access feature type of current feature
        # feature_type = attr_types[a]

        # if feature_type == 'continuous':
        #     # loop through all the neighborhood pairs
        #     for i, j in global_neighborhood_pairs:
        #         a_i = X_arr[i, a]
        #         a_j = X_arr[j, a]

        #         diff_a = np.abs(a_i - a_j)

        #         dist_a.append(diff_a)
        # else: # feature_type is categorical
        #     # loop through all the neighborhood pairs
        #     for i, j in global_neighborhood_pairs:
        #         a_i = X_arr[i, a]
        #         a_j = X_arr[j, a]

        #         diff_a = 1 if a_i != a_j else 0

        #         dist_a.append(diff_a)
            
        # # distance vector for attribute a -> ensure it is a np array
        # dist_a = np.array(dist_a)

        # *** vectorized version of commented out code above (dist_a computation)
        # access feature type of current feature
        feature_type = attr_types[a]

        if feature_type == "continuous":
            # difference between targets and neighbors in feature a (continuous)
            dist_a = np.abs(
                X_arr[global_neighborhood_pairs[:, 0], a] -
                X_arr[global_neighborhood_pairs[:, 1], a]
            )
        else:
            # difference between targets and neighbors in feature a (categorical)
            dist_a = (
                X_arr[global_neighborhood_pairs[:, 0], a] !=
                X_arr[global_neighborhood_pairs[:, 1], a]
            ).astype(int)
        # *** end of vectorized version for dist_a computation
        # print("Dist_a: \n", dist_a)
        # print("Dist_a shape:", dist_a.shape)

        # if there are no covariates, outcome difference is only regressed on attribute a difference
        if self._covariates is None:
            X_model = dist_a.reshape(-1, 1) # making sure X_model is 2D
        else:
            X_model = np.column_stack((dist_a, dist_covariates)) # if covariates exist, add them to matrix

        # add y-intercept term
        X_model = sm.add_constant(X_model)
        # print("X model: \n", X_model)
        # print("X_model shape:", X_model.shape)

        if self._class_type == "continuous":
            model = sm.OLS(dist_y, X_model).fit()

            beta_a = model.params[1]
            z_beta_a = model.tvalues[1] # coefficient of a's z-score (referred to as 'standardized beta' in original paper)
            pvalue = t.sf(z_beta_a, df=model.df_resid) # one-sided p-value; uses survival function to test if z_beta_a > 0 (alternative hypothesis), i.e. probability of z_beta_a being this large assuming null is true
        else: # outcome is binary
            model = sm.Logit(dist_y, X_model).fit(disp=False)

            beta_a = model.params[1]
            z_beta_a = model.tvalues[1] # coefficient of a's z-score (referred to as 'standardized beta' in original paper)
            pvalue = norm.sf(z_beta_a) # one-sided p-value

        return (beta_a, z_beta_a, pvalue)
    
    def summary(self, sort=True, feature_name=None, show_feature_type=False):
        """Provides a summary of the features with their importance scores, ranks, and feature types.
        Parameters
        ----------
        sort: bool, optional
            Whether to sort the features by importance. Default is True.
        feature_name: list of str or None, optional
            A list of feature names. If None, feature indices will be used. Default is None.
        show_feature_type: bool, optional
            Whether to display the type of the features. Default if False.

        Returns
        -------
        None
            Prints the summary of the features directly to the console.
        """

        data_type = [v[0] for v in self.attr.values()]
        id_order = self.top_features_ if sort else range(self._num_attributes)
        rank_dict = {feature: rank + 1 for rank, feature in enumerate(self.top_features_)}

        printed_name = feature_name if feature_name is not None else [str(i) for i in range(self._num_attributes)]

        # Process feature name length
        max_length = 40
        min_width = 15
        longest_name_length = max(len(name) for name in printed_name)
        if longest_name_length > max_length:
            printed_name = [
                name if len(name) <= max_length else name[:max_length - 3] + "..."
                for name in printed_name
            ]
            column_width = max_length+1 
        else:
            printed_name = printed_name
            column_width = max(longest_name_length+1, min_width)

        if show_feature_type:
            print(f"{'Feature name':<{column_width}}{'Feature importances':<23}{'Feature rank':<15}{'Raw Beta':<15}{'P-value':<15}{'Adj. P-value':<15}{'Feature type':<15}")
            for idx in id_order:
                print(f"{printed_name[idx]:<{column_width}}{self.feature_importances_[idx]:<23.8f}{rank_dict[idx]:<15}{self.beta_[idx]:<15.8f}{self.pvalues_[idx]:<15.4e}{self.pvalues_adj_[idx]:<15.4e}{data_type[idx]:<15}")

        else:
            print(f"{'Feature name':<{column_width}}{'Feature importances':<23}{'Feature rank':<15}{'Raw Beta':<15}{'P-value':<15}{'Adj. P-value':<15}")
            for idx in id_order:
                print(f"{printed_name[idx]:<{column_width}}{self.feature_importances_[idx]:<23.8f}{rank_dict[idx]:<15}{self.beta_[idx]:<15.8f}{self.pvalues_[idx]:<15.4e}{self.pvalues_adj_[idx]:<15.4e}")


