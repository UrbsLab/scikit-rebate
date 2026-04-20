Master status: [![Master Build Status](https://travis-ci.org/EpistasisLab/scikit-rebate.svg?branch=master)](https://travis-ci.org/EpistasisLab/scikit-rebate)
[![Master Code Health](https://landscape.io/github/EpistasisLab/scikit-rebate/master/landscape.svg?style=flat)](https://landscape.io/github/EpistasisLab/scikit-rebate/master)
[![Master Coverage Status](https://coveralls.io/repos/github/EpistasisLab/scikit-rebate/badge.svg?branch=master&service=github)](https://coveralls.io/github/EpistasisLab/scikit-rebate?branch=master)

Development status: [![Development Build Status](https://travis-ci.org/EpistasisLab/scikit-rebate.svg?branch=development)](https://travis-ci.org/EpistasisLab/scikit-rebate)
[![Development Code Health](https://landscape.io/github/EpistasisLab/scikit-rebate/development/landscape.svg?style=flat)](https://landscape.io/github/EpistasisLab/scikit-rebate/development)
[![Development Coverage Status](https://coveralls.io/repos/github/EpistasisLab/scikit-rebate/badge.svg?branch=development&service=github)](https://coveralls.io/github/EpistasisLab/scikit-rebate?branch=development)

Package information: ![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.6-blue.svg)
![License](https://img.shields.io/badge/license-MIT%20License-blue.svg)
[![PyPI version](https://badge.fury.io/py/skrebate.svg)](https://badge.fury.io/py/skrebate)

# scikit-rebate

This package includes a scikit-learn-compatible Python implementation of ReBATE, a suite of [Relief-based feature selection algorithms](<https://en.wikipedia.org/wiki/Relief_(feature_selection)>) for Machine Learning. These Relief-Based algorithms (RBAs) are designed for feature weighting/selection as part of a machine learning pipeline (supervised learning). Presently this includes the following core RBAs: ReliefF, SURF, SURF\*, MultiSURF\*, MultiSURF, SWRF\*, SWRF, MultiSWRF\*, MultiSWRF, MultiSWRFDB\*, MultiSWRFDB, and μ-Relief. Additionally, an implementation of the iterative TuRF mechanism and VLSRelief is included. **It is still under active development** and we encourage you to check back on this repository regularly for updates.

These algorithms offer a computationally efficient way to perform feature selection that is sensitive to feature interactions as well as simple univariate associations, unlike most currently available filter-based feature selection methods. The main benefit of Relief algorithms is that they identify feature interactions without having to exhaustively check every pairwise interaction, thus taking significantly less time than exhaustive pairwise search.

<!-- Certain algorithms require user specified run parameters (e.g. ReliefF requires the user to specify some 'k' number of nearest neighbors).  -->

Certain algorithms have run parameters that the user can specify, or if not specified, default to preset values (e.g. ReliefF’s parameter for ‘k’ number of nearest neighbors)

Relief algorithms are commonly applied to genetic analyses, where epistasis (i.e., feature interactions) is common. However, the algorithms implemented in this package can be applied to almost any supervised, structured data set and supports:

- Feature sets that are discrete/categorical, continuous-valued or a mix of both

- Data with missing values

- Binary endpoints (i.e., classification)

- Multi-class endpoints (i.e., classification)

- Continuous endpoints (i.e., regression)

Built into this code is a strategy to 'automatically' detect these relevant characteristics from the loaded data.

Of our two initial ReBATE software releases, this scikit-learn compatible version primarily focuses on ease of incorporation into a scikit learn analysis pipeline.
This code is most appropriate for scikit-learn users, Windows operating system users, beginners, or those looking for the most recent ReBATE developments.

An alternative 'stand-alone' version of [ReBATE](https://github.com/EpistasisLab/ReBATE) is also available that focuses on improving run-time with the use of Cython for optimization. This implementation also outputs feature names and associated feature scores as a text file by default.

## License

Please see the [repository license](https://github.com/EpistasisLab/scikit-rebate/blob/master/LICENSE) for the licensing and usage information for scikit-rebate.

Generally, we have licensed scikit-rebate to make it as widely usable as possible.

## Installation

scikit-rebate is built on top of the following existing Python packages:

- NumPy

- SciPy

- scikit-learn

All of the necessary Python packages can be installed via the [Anaconda Python distribution](https://www.continuum.io/downloads), which we strongly recommend that you use. We also strongly recommend that you use Python 3 over Python 2 if you're given the choice.

NumPy, SciPy, and scikit-learn can be installed in Anaconda via the command:

```
conda install numpy scipy scikit-learn
```

Once the prerequisites are installed, you should be able to install scikit-rebate with a pip command:

```
pip install skrebate
```

Please [file a new issue](https://github.com/EpistasisLab/scikit-rebate/issues/new) if you run into installation problems.

## Usage

### Basic Usage

To use an algorithm from ReBATE as a feature selection method:

```Python
# Import necessary packages
import pandas as pd
from skrebate import ReliefF

# Load the example dataset
genetic_data = pd.read_csv(
    './data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv')

# Separate the features and labels from the dataset
features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# Apply the ReliefF algorithm for feature selection
fs = ReliefF(discrete_threshold=10)
fs.fit(features, labels)

# Print out the results
feature_name = genetic_data.drop('class', axis=1).columns
fs.summary(feature_name=feature_name)

>>> Feature name   Feature importances    Feature rank
>>> P2             0.12330000             1
>>> P1             0.11892500             2
>>> N0             -0.00018125            3
>>> N10            -0.00075625            4
>>> N13            -0.00320625            5
>>> N14            -0.00402500            6
>>> N4             -0.00582500            7
>>> N1             -0.00595000            8
>>> N8             -0.00653750            9
>>> N12            -0.00696250            10
>>> N16            -0.00705000            11
>>> N17            -0.00740625            12
>>> N5             -0.00788750            13
>>> N11            -0.00822500            14
>>> N9             -0.00826250            15
>>> N2             -0.00871875            16
>>> N3             -0.00872500            17
>>> N7             -0.00991875            18
>>> N6             -0.01038750            19
>>> N15            -0.01044375            20
```

### Using as End-to-end Pipeline

We have designed the Relief-based algorithms to be integrated directly into scikit-learn machine learning workflows. For example, the ReliefF algorithm can be used as a feature selection step in a scikit-learn pipeline as follows.

```Python
# Import necessary packages
import pandas as pd
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the example dataset
genetic_data = pd.read_csv(
    './data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv')

# Separate the features and labels from the dataset
features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# Split the data to training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Make pipeline
clf = make_pipeline(
    ReliefF(n_features_to_select=2, discrete_threshold=10),
    RandomForestClassifier(n_estimators=100)
)

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model on testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
>>> Accuracy: 0.781
```

<!-- For more information on the Relief-based algorithms available in this package and how to use them, please refer to our [usage documentation](https://EpistasisLab.github.io/scikit-rebate/using/).

For updated documentation in this forked repository, please refer to this [updated usage documentation](https://urbslab.github.io/scikit-rebate/using/) -->

For more information on the Relief-based algorithms available in this package and how to use them, please refer to our [usage documentation](https://urbslab.github.io/scikit-rebate/using/).

## Contributing to scikit-rebate

We welcome you to [check the existing issues](https://github.com/EpistasisLab/scikit-rebate/issues/) for bugs or enhancements to work on. If you have an idea for an extension to scikit-rebate, please [file a new issue](https://github.com/EpistasisLab/scikit-rebate/issues/new) so we can discuss it.

Please refer to our [contribution guidelines](https://EpistasisLab.github.io/scikit-rebate/contributing/) prior to working on a new feature or bug fix.

## Citing scikit-rebate

If you use scikit-rebate in a scientific publication, please consider citing the following paper:

Ryan J. Urbanowicz, Randal S. Olson, Peter Schmitt, Melissa Meeker, Jason H. Moore (2018). Benchmarking Relief-Based Feature Selection Methods for Bioinformatics Data Mining. _Journal of Biomedical Informatics_, 85, 168-188. DOI: [10.1016/j.jbi.2018.07.015](https://doi.org/10.1016/j.jbi.2018.07.015)

### BibTeX entry:

```bibtex
@article{Urbanowicz2018Benchmarking,
  author    = {Urbanowicz, Ryan J. and Olson, Randal S. and Schmitt, Peter and Meeker, Melissa and Moore, Jason H.},
  title     = {Benchmarking Relief-Based Feature Selection Methods for Bioinformatics Data Mining},
  journal   = {Journal of Biomedical Informatics},
  volume    = {85},
  pages     = {168--188},
  year      = {2018},
  doi       = {10.1016/j.jbi.2018.07.015},
  url       = {https://doi.org/10.1016/j.jbi.2018.07.015}
}
```
