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

    def __init__(self, regression_type=None, relief_object=None, padj_method="fdr_bh", n_jobs=1):
        """Sets up NPDR to perform feature selection
        Parameters
        ----------
        regression_type: str (default: None)
            Type of regression to run on the outcome variable. Options are "linear" and "logistic". If input is not provided
            this is automatically determined based on the number of unique values in the outcome variable "y".
        relief_object: object (default: None)
            A core Relief-based algorithm (RBA) that has a _find_neighbors function. Used to identify neighbor instances for each target.
        padj_method: str (default: "fdr_bh")
            Method used to adjust p-values for multiple testing. 
        n_jobs: int (default: 1)
            The number of cores to dedicate to computing the scores with joblib.
            Assigning this parameter to -1 will dedicate as many cores as are available on your system.
            We recommend setting this parameter to -1 to speed up the algorithm as much as possible.
        """

        self.regression_type = regression_type
        self.relief_object = (MultiSWRFDB() if relief_object is None else relief_object)
        self.padj_method = padj_method
        self.n_jobs = n_jobs

        self._validate_params()

    def _validate_params(self):
        if self.regression_type is not None and self.regression_type not in ("linear", "logistic"):
            raise ValueError(
                "regression_type must be either 'linear' or 'logistic'"
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
        if self.regression_type == "linear" and isinstance(self.relief_object, (ReliefF, MuRelief)):
            raise ValueError(
                "If regression_type is 'linear', a radius-based relief_object must be used"
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

        # Number of unique outcome values (used to determine type of regression to run if user doesn't specify)
        self._label_list = list(set(self._y))

        if len(self._label_list) == 1:
            raise ValueError('All labels are of the same class.')
        # if regression_type is provided:
        if self.regression_type is not None:
            if self.regression_type == 'logistic' and len(self._label_list) != 2:
                raise ValueError("Specified logistic regression, but the number of unique labels is not 2. Multiclass logistic regression is not supported.") 
            # self._class_type = self.label_type # analogous to self.regression_type
        # if regression_type is NOT provided, auto-detect:
        else:
            if len(self._label_list) == 2:
                self.regression_type = 'logistic'
            elif len(self._label_list) <= 10:
                self.regression_type = 'linear'
                warnings.warn(
                    "Detected between 3 and 10 unique y-values. Multiclass data is not supported; applying linear regression.",
                    UserWarning,
                )
            else:
                self.regression_type = 'linear'

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
        # Examines each feature and applies categorical_threshold to determine variable type.
        # Manually pre-setting categorical_features to None & categorical_threshold to 10 before application of _get_attribute_info
        self.categorical_features = None
        self.categorical_threshold = 10
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

        return self
