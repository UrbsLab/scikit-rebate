# from sklearn.base import BaseEstimator
from .npdr import NPDR
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
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, ElasticNet, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import warnings
import time

class ElasticNetNPDR(NPDR):

    """Variant of NPDR that passes all predictors into a single regression model.
    Utilizes regularization to extract and highlight the most relevant features. Unlike NPDR, does not support covariates.
    Based on the description of Regularized NPDR in:
    T.T. Le, B.A. Dawkins, and B.A. McKinney
    Nearest-neighbor Projected-Distance Regression (NPDR) for detecting network interactions with adjustments for multiple tests and confounding
    """

    def __init__(self, n_features_to_select=10, label_type=None, relief_object=None, categorical_threshold=10, 
                 categorical_features=None, l1_ratio=0.5, alpha="auto", n_jobs=1):
        """Sets up ElasticNetNPDR to perform feature selection
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
        categorical_threshold: int (default: 10)
            Value used to determine if a feature is categorical/discrete or continuous.
            If the number of unique values in a feature is > categorical_threshold, then it is
            considered continuous, and categorical otherwise.
        categorical_features: list (default: None)
            List of index columns indicating features to be treated as categorical.
            If set to None, the features will be automatically classified based on the categorical_threshold.
        l1_ratio: float (default: 0.5)
            The combination of L1 and L2 penalty used. l1_ratio=1 applies the L1 penalty, 
            l1_ratio=0 applies the L2 penalty, and 0 < l1_ratio < 1 applies a combination of L1 and L2 penalty.
        alpha: str/float (default: "auto")
            Constant that multiplies the penalty terms. If "auto", alpha is selected through cross validation.
        n_jobs: int (default: 1)
            The number of cores to dedicate to computing the scores with joblib.
            Assigning this parameter to -1 will dedicate as many cores as are available on your system.
            We recommend setting this parameter to -1 to speed up the algorithm as much as possible.
        """
        
        self.n_features_to_select = n_features_to_select
        self.label_type = label_type
        self.relief_object = (MultiSWRFDB() if relief_object is None else relief_object)
        self.categorical_threshold = categorical_threshold
        self.categorical_features = categorical_features
        self.l1_ratio = l1_ratio
        self.alpha = alpha
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
        
        if not isinstance(self.categorical_threshold, int):
            raise TypeError("categorical_threshold must be an integer")
        if self.categorical_threshold < 2:
            raise ValueError("categorical_threshold must be >= 2")
        
        if self.categorical_features is not None and not isinstance(self.categorical_features, list):
            raise TypeError("categorical_features must be a list of feature indices or None")
        
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1")

        if self.alpha != "auto":
            if not isinstance(self.alpha, (int, float)):
                raise ValueError("alpha must be 'auto' or a positive numeric value")
            if self.alpha <= 0:
                raise ValueError("alpha must be positive")

        if not isinstance(self.n_jobs, int):
            raise TypeError("n_jobs must be an integer")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be a positive integer or -1")
        
    def fit(self, X, y):
        """Scikit-learn required: Computes the feature importance scores (beta's) from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training outcomes

        Returns
        -------
        Copy of the ElasticNetNPDR instance
        """

        self._X = X  # matrix of predictive variables ('independent variables')
        self._y = y  # vector of values for outcome variable ('dependent variable')

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
        
        # ensure y is np array
        y_arr = np.asarray(self._y)

        pairs = np.asarray(global_neighborhood_pairs)

        # computing distance vector for y (outcome)
        if self._class_type == "continuous":
            # difference between targets and neighbors in outcome (continuous)
            dist_y = np.abs(y_arr[pairs[:, 0]] - y_arr[pairs[:, 1]])
        else:
            # difference between targets and neighbors in outcome (binary, different = 1 equal = 0)
            dist_y = (y_arr[pairs[:, 0]] != y_arr[pairs[:, 1]]).astype(int)
        
        # will be used to identify feature type for each feature
        attr_types = [value[0] for value in self.attr.values()]
        # ensure X is np array
        X_arr = np.asarray(self._X)

        # computing distance vector for X (features)
        dist_X = np.empty((len(pairs), X_arr.shape[1]))

        for feat_idx, feat_type in enumerate(attr_types):
            if feat_type == "continuous":
                # difference between targets and neighbors in current feature (continuous)
                dist_X[:, feat_idx] = np.abs(
                    X_arr[pairs[:, 0], feat_idx] -
                    X_arr[pairs[:, 1], feat_idx]
                )
            else:
                # difference between targets and neighbors in current feature (categorical, different = 1 equal = 0)
                dist_X[:, feat_idx] = (
                    X_arr[pairs[:, 0], feat_idx] !=
                    X_arr[pairs[:, 1], feat_idx]
                ).astype(int)
        
        # since the final feature importances are raw betas (not z-scores/t-scores), must standardize dist_X for raw betas to be comparable
        self.scaler_ = StandardScaler()
        dist_X = self.scaler_.fit_transform(dist_X)

        # creating regression model with all predictive features, returning coefficients (feature importances)
        feature_importances = self._create_regression_model(dist_X, dist_y)

        self.feature_importances_ = feature_importances

        # sorting feature indices based on feature importance score
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        return self
    
    # def _create_regression_model(self, dist_X, dist_y):
    #     # if self.alpha = "auto", use CV to select alpha for model
    #     if self.alpha == "auto":
    #         if self._class_type == "continuous": # if continuous, use ElasticNetCV package
    #             cv_model = ElasticNetCV(
    #                 l1_ratio=self.l1_ratio,
    #                 cv=5
    #             )

    #             cv_model.fit(dist_X, dist_y)

    #             alpha_selected = cv_model.alpha_ # alpha selected after CV
    #         else: # if binary, use LogisticRegressionCV with penalty="elasticnet" and solver="saga"
    #             cv_model = LogisticRegressionCV(
    #                 penalty="elasticnet",
    #                 solver="saga",
    #                 l1_ratios=[self.l1_ratio],
    #                 cv=5,
    #                 max_iter=5000,
    #                 random_state=42
    #             )

    #             cv_model.fit(dist_X, dist_y)

    #             alpha_selected = 1 / cv_model.C_[0] # since C = 1/alpha (1/lambda), extract best alpha like this
    #     # else, use positive numeric alpha provided by user
    #     else:
    #         alpha_selected = self.alpha

    #     # for user to see what alpha was selected for the model
    #     self.alpha_ = alpha_selected 

    #     # # add y-intercept term for statsmodels
    #     # X_model = sm.add_constant(dist_X)

    #     if self._class_type == "continuous":
    #         # model = sm.GLM(
    #         #     dist_y,
    #         #     X_model,
    #         #     family=sm.families.Gaussian()
    #         # ).fit_regularized(
    #         #     alpha=alpha_selected,
    #         #     L1_wt=self.l1_ratio
    #         # )

    #         # # exclude intercept
    #         # coefficients = model.params[1:]
    #         model = ElasticNet(
    #             l1_ratio=self.l1_ratio,
    #             alpha=alpha_selected
    #         ).fit(dist_X, dist_y)

    #         # ElasticNet stores intercept separately, so this is just the features
    #         coefficients = model.coef_

    #     else:  # outcome is binary
    #         # model = sm.GLM(
    #         #     dist_y,
    #         #     X_model,
    #         #     family=sm.families.Binomial()
    #         # ).fit_regularized(
    #         #     alpha=alpha_selected,
    #         #     L1_wt=self.l1_ratio
    #         # )

    #         # # exclude intercept
    #         # coefficients = model.params[1:]
    #         model = LogisticRegression(
    #             penalty="elasticnet",
    #             solver="saga",
    #             l1_ratio=self.l1_ratio,
    #             C=1.0/alpha_selected,
    #             max_iter=5000,
    #             random_state=42
    #         ).fit(dist_X, dist_y)

    #         # excludes intercept; coef_ here is 2D so turning into 1D
    #         coefficients = model.coef_[0]

    #     return coefficients

    # ** _create_regression_model with non-negativity constraints on coefficients
    def _create_regression_model(self, dist_X, dist_y):
        # if self.alpha = "auto", use CV to select alpha for model
        if self.alpha == "auto":
            alpha_selected = self._select_alpha_cv(
                dist_X,
                dist_y
            )
        # else, use positive numeric alpha provided by user
        else:
            alpha_selected = self.alpha

        # for user to see what alpha was selected for the model
        self.alpha_ = alpha_selected

        # fit regularized model to find optimized coefficients (feature importances)
        # use selected alpha value, l1_ratio as indicated in constructor
        _, coefficients = self._fit_elastic_net(dist_X, dist_y, alpha_selected)

        return coefficients
    
    def _select_alpha_cv(self, X, y):
        # 50 candidate regularization strengths (from 10^-5 to 10^2)
        alpha_grid = np.logspace(-5, 2, 50)

        kfold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_alpha)(
                alpha,
                X,
                y,
                kfold
            )
            for alpha in alpha_grid
        )

        alpha_scores = dict(results)

        # Select lowest validation loss
        best_alpha = min(
            alpha_scores,
            key=alpha_scores.get
        )

        return best_alpha
    
    def _evaluate_alpha(self, alpha, X, y, kfold):
        """Helper function of _select_alpha_cv to test a given alpha value"""
        fold_scores = []

        for train_idx, val_idx in kfold.split(X):

            X_train = X[train_idx]
            X_val = X[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]

            intercept, coefficients = self._fit_elastic_net(
                X_train,
                y_train,
                alpha
            )

            predictions = self._predict_elastic_net(
                X_val,
                intercept,
                coefficients
            )

            if self._class_type == "continuous":
                score = mean_squared_error(
                    y_val,
                    predictions
                )
            else:
                score = log_loss(
                    y_val,
                    predictions
                )

            fold_scores.append(score)

        return alpha, np.mean(fold_scores)

    def _fit_elastic_net(self, X, y, alpha):
        """Function to fit and optimize coefficient values given input settings."""
        n_features = X.shape[1]

        # initial coefficient values, will be optimized using designated method
        initial_params = np.zeros(n_features + 1)

        # objective function that will optimize coefficient values
        if self._class_type == "continuous":
            objective_function = self.elastic_net_gaussian_objective
        else: # outcome is binary
            objective_function = self.elastic_net_logistic_objective

        # compute coefficients, with a non-negativity constraint (no coefficient can have negative value)
        result = minimize(
            objective_function,
            initial_params,
            args=(
                X,
                y,
                alpha,
                self.l1_ratio
            ),
            method="L-BFGS-B",
            bounds=[
                (None, None)   # intercept
            ] + [
                (0, None)      # beta >= 0
            ] * n_features,
            options={
                "maxiter": 5000
            }
        )

        if not result.success:
            raise RuntimeError(result.message)

        # intercept, feature coefficients
        return result.x[0], result.x[1:] 
    
    def _predict_elastic_net(self, X, intercept, coefficients):
        """Function """
        logits = intercept + X @ coefficients

        if self._class_type == "continuous":
            return logits

        else:
            return expit(logits)

    @staticmethod
    def elastic_net_logistic_objective(params, X, y, alpha, l1_ratio):
        """
        params[0] = intercept
        params[1:] = coefficients
        """

        intercept = params[0]
        beta = params[1:]

        # logistic negative log likelihood
        logits = intercept + X @ beta
        probs = expit(logits)

        eps = 1e-12
        nll = -np.mean(
            y * np.log(probs + eps)
            + (1-y) * np.log(1-probs + eps)
        )

        # Elastic net penalty (do not penalize intercept)
        l1 = l1_ratio * np.sum(np.abs(beta))
        l2 = (1-l1_ratio) * np.sum(beta**2) / 2

        penalty = alpha * (l1 + l2)

        return nll + penalty
    
    @staticmethod
    def elastic_net_gaussian_objective(params, X, y, alpha, l1_ratio):
        """
        params[0] = intercept
        params[1:] = coefficients
        """

        intercept = params[0]
        beta = params[1:]

        # prediction
        y_pred = intercept + X @ beta

        # Gaussian negative log likelihood (MSE)
        mse = np.sum((y - y_pred)**2) / (2 * len(y))

        # Elastic Net penalty
        l1 = l1_ratio * np.sum(np.abs(beta))
        l2 = (1 - l1_ratio) * np.sum(beta**2) / 2

        penalty = alpha * (l1 + l2)

        return mse + penalty
    
    def summary(self, sort=True, feature_name=None, show_feature_type=False):
        """Provides a summary of the features with their importance scores, ranks, and feature types.
        Parameters
        ----------
        sort: bool, optional
            Whether to sort the features by importance. Default is True.
        feature_name: list of str or None, optional
            A list of feature names. If None, feature indicies will be used. Default is None.
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
            print(f"{'Feature name':<{column_width}}{'Feature importances':<23}{'Feature rank':<15}{'Feature type':<15}")
            for idx in id_order:
                print(f"{printed_name[idx]:<{column_width}}{self.feature_importances_[idx]:<23.8f}{rank_dict[idx]:<15}{data_type[idx]:<15}")

        else:
            print(f"{'Feature name':<{column_width}}{'Feature importances':<23}{'Feature rank':<15}")
            for idx in id_order:
                print(f"{printed_name[idx]:<{column_width}}{self.feature_importances_[idx]:<23.8f}{rank_dict[idx]:<15}")
