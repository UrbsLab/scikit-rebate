# from sklearn.base import BaseEstimator
from .relieff import ReliefF
from .surf import SURF
from .surfstar import SURFstar
from .multisurf import MultiSURF
from .multisurfstar import MultiSURFstar
from .baseswrf import SWRF, SWRFstar, MultiSWRF, MultiSWRFstar, MultiSWRFDB, MultiSWRFDBstar
from .murelief import MuRelief
import numpy as np
import warnings
import time

class NPDR(ReliefF):

    """Feature selection algorithm designed for continuous outcome data (also handles binary outcomes).
    Based on the NPDR algorithm as introduced in:
    T.T. Le, B.A. Dawkins, and B.A. McKinney
    Nearest-neighbor Projected-Distance Regression (NPDR) for detecting network interactions with adjustments for multiple tests and confounding
    """

    def __init__(self, label_type=None, relief_object=None, padj_method="fdr_bh", categorical_threshold=10, 
                 categorical_features=None, categorical_covariates=None, n_jobs=1):
        """Sets up NPDR to perform feature selection
        Parameters
        ----------
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
            considered continuous, or categorical otherwise.
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

        self.label_type = label_type
        self.relief_object = (MultiSWRFDB() if relief_object is None else relief_object)
        self.padj_method = padj_method
        self.categorical_threshold = categorical_threshold
        self.categorical_features = categorical_features
        self.categorical_covariates = categorical_covariates
        self.n_jobs = n_jobs

        self._validate_params()

    def _validate_params(self):
        if self.label_type is not None and self.label_type not in ("binary", "continuous"):
            raise ValueError(
                "label_type must be either 'binary' or 'continuous'"
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
        if self.label_type == "continuous" and isinstance(self.relief_object, (ReliefF, MuRelief)):
            raise ValueError(
                "If label_type is 'continuous', a radius-based relief_object must be used"
            )

        allowed_methods = (
            "bonferroni",
            "sidak",
            "holm",
            "holm-sidak",
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
            elif len(self._label_list) <= 10:
                self._class_type = 'continuous'
                warnings.warn(
                    "Detected between 3 and 10 unique outcome values. Multiclass data is not supported; encoding as continuous outcome.",
                    UserWarning,
                )
            else:
                self._class_type = 'continuous'

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

        # list to contain all neighbor pairs across all target instances
        global_neighborhood_pairs = [] 

        # Checking what type of instance self.relief_object is
        if isinstance(self.relief_object, (ReliefF, MuRelief, MultiSURF)):
            neighbor_list = [self.relief_object._find_neighbors(datalen) for datalen in range(self._datalen)]

            # creating pairs for global neighborhood, each of the tuple form (target_idx, neighbor_idx)
            for target_idx, neighbors in enumerate(neighbor_list):
                for neighbor_idx in neighbors:
                    global_neighborhood_pairs.append((target_idx, neighbor_idx))

        elif isinstance(self.relief_object, (MultiSURFstar)):
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
        
        elif isinstance(self.relief_object, (SURF)):
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
        
        elif isinstance(self.relief_object, (SURFstar)):
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

        elif isinstance(self.relief_object, (SWRF, MultiSWRF, MultiSWRFDB)):
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
        
        elif isinstance(self.relief_object, (SWRFstar, MultiSWRFstar, MultiSWRFDBstar)):
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
        # global neighborhood computation completed

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

        # distance vector for y and covariates 
        dist_y = []
        dist_covariates = []

        for i, j in global_neighborhood_pairs:
            # value target i has for y
            y_i = y_arr[i]
            # value neighbor j has for y
            y_j = y_arr[j]

            if self._class_type == "continuous":
                # difference between target and neighbor in outcome (continuous)
                diff_y = np.abs(y_i - y_j)
            else:
                # difference between target and neighbor in outcome (binary, different = 1 equal = 0)
                diff_y = 1 if y_i != y_j else 0

            # add distance in y between (target i and neighbor j) to y distance vector
            dist_y.append(diff_y)

            # if covariates is not None, add to distance vector/matrix for covariates
            if self._covariates is not None:
                # list to contain differences between current target and neighbor for all covariates (a "row" of differences, columns = # of covariates)
                cov_differences = []
                for cov_idx in range(covariates_arr.shape[1]):
                    # values for current covariate 'cov' for target i and neighbor j
                    cov_i = covariates_arr[i, cov_idx]
                    cov_j = covariates_arr[j, cov_idx]

                    if covariate_types[cov_idx] == "continuous":
                        # difference between target and neighbor in current covariate (continuous)
                        diff_cov = np.abs(cov_i - cov_j)
                    else:
                        # difference between target and neighbor in current covariate (categorical, different = 1 equal = 0)
                        diff_cov = 1 if cov_i != cov_j else 0
                    
                    # add difference for this covariate to the list of all covariate differences for current i and j
                    cov_differences.append(diff_cov)

                dist_covariates.append(cov_differences) 
        
        dist_y = np.array(dist_y)
        dist_covariates = np.array(dist_covariates)



        return self
