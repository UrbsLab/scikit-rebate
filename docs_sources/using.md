# Using skrebate
We have designed the Relief algorithms to be integrated directly into scikit-learn machine learning workflows. Below, we provide code samples showing how the various Relief algorithms can be used as feature selection methods in scikit-learn pipelines.

For details on the algorithmic differences between the various Relief algorithms, please refer to [this research paper](https://arxiv.org/abs/1711.08477).


## Using the Core Algorithms

### ReliefF

ReliefF is the most basic of the Relief-based feature selection algorithms, and it requires you to specify the number of nearest neighbors to consider in the scoring algorithm. The parameters for the ReliefF algorithm are as follows:

| Parameter | Valid values | Default value | Effect |
|-----------|--------------|---------------|--------|
| `n_features_to_select`| Any positive integer or float       | 10      | The number of best features to retain after the feature selection process. The "best" features are the highest-scored features according to the ReliefF scoring process. |
| `n_neighbors`         | Any positive integer               | 100     | The number of neighbors to consider when assigning feature importance scores. If a float number is provided, that percentage of training samples is used as the number of neighbors. More neighbors result in more accurate scores, but takes longer. |
| `categorical_features`| A list of integers                 | `None`    | List of index columns indicating features to be treated as categorical. If set to None, the features will be automatically classified based on the `discrete_threshold` below. |
| `discrete_threshold`  | Any positive integer               | 2       | Value used to determine if a feature is discrete or continuous. If the number of unique levels in a feature is > `discrete_threshold`, then it is considered continuous; otherwise, it is discrete. |
| `n_jobs`              | Any positive integer or -1         | 1       | The number cores to dedicate to running the algorithm in parallel with joblib. Set to -1 to use all available cores.                                      |
| `label_type`          | Choose from `None`, `'binary'`, `'multiclass'`, `'continuous'` | `None` | With default value as `None`, the function automatically infers the label type based on the number of unique labels: 2 for `'binary'`, 3-10 for `'multiclass'`, and >10 for `'continuous'`.


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

### SURF

SURF, SURF*, MultiSURF, and MultiSURF* are all extensions to the ReliefF algorithm that automatically determine the ideal number of neighbors to consider when scoring the features.

| Parameter | Valid values | Default value | Effect |
|-----------|--------------|---------------|--------|
| `n_features_to_select`| Any positive integer or float       | 10      | The number of best features to retain after the feature selection process. The "best" features are the highest-scored features according to the ReliefF scoring process. |
| `categorical_features`| A list of integers                 | `None`    | List of index columns indicating features to be treated as categorical. If set to None, the features will be automatically classified based on the `discrete_threshold` below. |
| `discrete_threshold`  | Any positive integer               | 2       | Value used to determine if a feature is discrete or continuous. If the number of unique levels in a feature is > `discrete_threshold`, then it is considered continuous; otherwise, it is discrete. |
| `n_jobs`              | Any positive integer or -1         | 1       | The number cores to dedicate to running the algorithm in parallel with joblib. Set to -1 to use all available cores.                                      |
| `label_type`          | Choose from `None`, `'binary'`, `'multiclass'`, `'continuous'` | `None` | With default value as `None`, the function automatically infers the label type based on the number of unique labels: 2 for `'binary'`, 3-10 for `'multiclass'`, and >10 for `'continuous'`.


```python
# Import necessary packages
import pandas as pd
from skrebate import SURF

# Load the example dataset
genetic_data = pd.read_csv(
    './data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv')

# Separate the features and labels from the dataset
features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# Apply the ReliefF algorithm for feature selection
fs = SURF(discrete_threshold=10)
fs.fit(features, labels)

# Print out the results
feature_name = genetic_data.drop('class', axis=1).columns
fs.summary(feature_name=feature_name)

>>> Feature name   Feature importances    Feature rank   
>>> P1             0.06157283             1              
>>> P2             0.06155927             2              
>>> N10            -0.00049277            3              
>>> N0             -0.00059267            4              
>>> N13            -0.00132625            5              
>>> N5             -0.00222361            6              
>>> N16            -0.00256536            7              
>>> N1             -0.00269856            8              
>>> N8             -0.00300784            9              
>>> N14            -0.00301537            10             
>>> N9             -0.00302025            11             
>>> N11            -0.00361912            12             
>>> N12            -0.00419101            13             
>>> N7             -0.00423327            14             
>>> N2             -0.00453031            15             
>>> N4             -0.00455866            16             
>>> N15            -0.00468981            17             
>>> N17            -0.00504565            18             
>>> N6             -0.00590625            19             
>>> N3             -0.00634880            20       
```

### SURF*

| Parameter | Valid values | Default value | Effect |
|-----------|--------------|---------------|--------|
| `n_features_to_select`| Any positive integer or float       | 10      | The number of best features to retain after the feature selection process. The "best" features are the highest-scored features according to the ReliefF scoring process. |
| `categorical_features`| A list of integers                 | `None`    | List of index columns indicating features to be treated as categorical. If set to None, the features will be automatically classified based on the `discrete_threshold` below. |
| `discrete_threshold`  | Any positive integer               | 2       | Value used to determine if a feature is discrete or continuous. If the number of unique levels in a feature is > `discrete_threshold`, then it is considered continuous; otherwise, it is discrete. |
| `n_jobs`              | Any positive integer or -1         | 1       | The number cores to dedicate to running the algorithm in parallel with joblib. Set to -1 to use all available cores.                                      |
| `label_type`          | Choose from `None`, `'binary'`, `'multiclass'`, `'continuous'` | `None` | With default value as `None`, the function automatically infers the label type based on the number of unique labels: 2 for `'binary'`, 3-10 for `'multiclass'`, and >10 for `'continuous'`.


```python
# Import necessary packages
import pandas as pd
from skrebate import SURFstar

# Load the example dataset
genetic_data = pd.read_csv(
    './data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv')

# Separate the features and labels from the dataset
features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# Apply the ReliefF algorithm for feature selection
fs = SURFstar(discrete_threshold=10)
fs.fit(features, labels)

# Print out the results
feature_name = genetic_data.drop('class', axis=1).columns
fs.summary(feature_name=feature_name)

>>> Feature name   Feature importances    Feature rank   
>>> P2             0.12864101             1              
>>> P1             0.12792470             2              
>>> N10            -0.00106372            3              
>>> N0             -0.00136069            4              
>>> N13            -0.00307977            5              
>>> N5             -0.00538099            6              
>>> N1             -0.00616617            7              
>>> N16            -0.00650725            8              
>>> N14            -0.00663881            9              
>>> N8             -0.00690113            10             
>>> N7             -0.00765252            11             
>>> N12            -0.00810566            12             
>>> N9             -0.00878649            13             
>>> N4             -0.00882028            14             
>>> N11            -0.00897106            15             
>>> N2             -0.00911812            16             
>>> N15            -0.01001887            17             
>>> N17            -0.01019251            18             
>>> N6             -0.01174531            19             
>>> N3             -0.01225238            20      
```

### MultiSURF
| Parameter | Valid values | Default value | Effect |
|-----------|--------------|---------------|--------|
| `n_features_to_select`| Any positive integer or float       | 10      | The number of best features to retain after the feature selection process. The "best" features are the highest-scored features according to the ReliefF scoring process. |
| `categorical_features`| A list of integers                 | `None`    | List of index columns indicating features to be treated as categorical. If set to None, the features will be automatically classified based on the `discrete_threshold` below. |
| `discrete_threshold`  | Any positive integer               | 2       | Value used to determine if a feature is discrete or continuous. If the number of unique levels in a feature is > `discrete_threshold`, then it is considered continuous; otherwise, it is discrete. |
| `n_jobs`              | Any positive integer or -1         | 1       | The number cores to dedicate to running the algorithm in parallel with joblib. Set to -1 to use all available cores.                                      |
| `label_type`          | Choose from `None`, `'binary'`, `'multiclass'`, `'continuous'` | `None` | With default value as `None`, the function automatically infers the label type based on the number of unique labels: 2 for `'binary'`, 3-10 for `'multiclass'`, and >10 for `'continuous'`.

```python
# Import necessary packages
import pandas as pd
from skrebate import MultiSURF

# Load the example dataset
genetic_data = pd.read_csv(
    './data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv')

# Separate the features and labels from the dataset
features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# Apply the ReliefF algorithm for feature selection
fs = MultiSURF(discrete_threshold=10)
fs.fit(features, labels)

# Print out the results
feature_name = genetic_data.drop('class', axis=1).columns
fs.summary(feature_name=feature_name)

>>> Feature name   Feature importances    Feature rank   
>>> P2             0.08845469             1              
>>> P1             0.08807613             2              
>>> N0             -0.00049561            3              
>>> N10            -0.00056682            4              
>>> N13            -0.00212670            5              
>>> N14            -0.00467041            6              
>>> N8             -0.00613933            7              
>>> N9             -0.00624673            8              
>>> N1             -0.00625501            9              
>>> N12            -0.00658989            10             
>>> N16            -0.00666320            11             
>>> N11            -0.00764327            12             
>>> N5             -0.00822562            13             
>>> N4             -0.00838633            14             
>>> N15            -0.00857736            15             
>>> N2             -0.00903556            16             
>>> N6             -0.00912725            17             
>>> N7             -0.00916942            18             
>>> N3             -0.00931820            19             
>>> N17            -0.00978209            20         
```

### MultiSURF*
| Parameter | Valid values | Default value | Effect |
|-----------|--------------|---------------|--------|
| `n_features_to_select`| Any positive integer or float       | 10      | The number of best features to retain after the feature selection process. The "best" features are the highest-scored features according to the ReliefF scoring process. |
| `categorical_features`| A list of integers                 | `None`    | List of index columns indicating features to be treated as categorical. If set to None, the features will be automatically classified based on the `discrete_threshold` below. |
| `discrete_threshold`  | Any positive integer               | 2       | Value used to determine if a feature is discrete or continuous. If the number of unique levels in a feature is > `discrete_threshold`, then it is considered continuous; otherwise, it is discrete. |
| `n_jobs`              | Any positive integer or -1         | 1       | The number cores to dedicate to running the algorithm in parallel with joblib. Set to -1 to use all available cores.                                      |
| `label_type`          | Choose from `None`, `'binary'`, `'multiclass'`, `'continuous'` | `None` | With default value as `None`, the function automatically infers the label type based on the number of unique labels: 2 for `'binary'`, 3-10 for `'multiclass'`, and >10 for `'continuous'`.

```python
# Import necessary packages
import pandas as pd
from skrebate import MultiSURFstar

# Load the example dataset
genetic_data = pd.read_csv(
    './data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv')

# Separate the features and labels from the dataset
features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# Apply the algorithm for feature selection
fs = MultiSURFstar(discrete_threshold=10)
fs.fit(features, labels)

# Print out the results
feature_name = genetic_data.drop('class', axis=1).columns
fs.summary(feature_name=feature_name)

>>> Feature name   Feature importances    Feature rank   
>>> P2             0.17498272             1              
>>> P1             0.17344628             2              
>>> N10            -0.00179863            3              
>>> N0             -0.00181761            4              
>>> N13            -0.00553229            5              
>>> N14            -0.01039264            6              
>>> N8             -0.01294620            7              
>>> N12            -0.01297249            8              
>>> N5             -0.01327019            9              
>>> N1             -0.01362589            10             
>>> N16            -0.01397024            11             
>>> N9             -0.01406904            12             
>>> N11            -0.01442700            13             
>>> N7             -0.01500246            14             
>>> N4             -0.01541660            15             
>>> N15            -0.01603204            16             
>>> N2             -0.01606567            17             
>>> N3             -0.01765580            18             
>>> N17            -0.01826211            19             
>>> N6             -0.01848672            20             
```

## Using as End-to-end pipline

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


## TuRF wrapper for larger feature set
TURF advances the feature selection process from a single round to a multi-round process, and can be used in conjunction with any of the Relief-based algorithms. TURF begins with all of the features in the first round, scores them using one of the Relief-based algorithms, then eliminates a portion of them that have the worst scores. With this reduced feature set, TURF again scores the remaining features and eliminates a portion of the worst-scoring features. This process is repeated until a predefined number of features remain or some maximum number of iterations have completed. Presently, there are two ways to run the 'TuRF' iterative feature selection wrapper around any of the given core Relief-based algorithm in scikit-rebate. First, there is a custom TuRF implementation, hard coded into scikit-rebate designed to operate in the same way as specified in the original TuRF paper.  The second, uses the [Recursive Feature Elimination](http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination), as [implemented](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) in scikit-learn. These approaches are similar but not equivalent. We recommend using built in scikit-rebate TuRF. Examples for running TuRF using either approach are given below.


### TuRF implemented in scikit-rebate
With this TuRF implementation, the (pct) parameter inversely determines the number of TuRF scoring iterations (i.e 1/pct), and pct also determines the percent of features eliminated from scoring each iteration. The n\_features\_to\_select parameter simply determines the number of top scoring features to pass onto the pipeline. This TuRF approach should be used to most closely follow the original TuRF description, as well as to be able to obtain individual feature scores following the completion of TuRF. This method also keeps information about when features were dropped from consideration during progressive TuRF iterations. It does this by assigning 'removed' features a token score that simply indicates which iteration the feature was removed from scoring. All features removed from scoring during TuRF will be assigned a score lower than the lowest feature score in the final feature set.  All features removed at the same time are assigned the same discounted token feature score. This is particularly important when accessing the feature scores as described later. For an example of how to use scikit-rebate TuRF in a scikit learn pipeline see below.
```python
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import TuRF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values
headers = list(genetic_data.drop("class", axis=1))

clf = make_pipeline(TuRF(core_algorithm="ReliefF", n_features_to_select=2, pct=0.5),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, features, labels, fit_params={'turf__headers': headers})))
>>> 0.795
```

### TuRF via RFE
With this strategy for running TuRF the main difference is that the number of TuRF iterations is not controlled by the pct parameter, rather, iterations run until the specified number of n\_features\_to\_select have been reached. Each iteration, the 'step' parameter controls the number of features removed, either as a percent between 0 and 1 or an integer count of features to remove each iteration. One critical shortcoming of this approach is that there is no way to obtain the individual feature scores when using RFE to do 'TuRF' scoring. See [Recursive Feature Elimination](http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination), for more details.

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

clf = make_pipeline(RFE(ReliefF(), n_features_to_select=2, step = 0.5),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, features, labels)))
>>> 0.795
```
## How to Set Up Feature Type
### manually
### automatic threshold

## Acquiring feature importance scores

In many cases, it may be useful to compute feature importance scores without actually performing feature selection. We have made it possible to access all Relief-based algorithm's scores via the `feature_importances_` attribute. Below are code examples showing how to access the scores from the any core Relief-based algorithm as well as from TuRF in combination with a Relief-based algorithm. The first example illustrates how scores may be obtained from ReliefF, adding a split of the loaded data into training and testing since we are not running ReliefF as part of a scikit pipeline like above.  

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.model_selection import train_test_split

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# Make sure to compute the feature importance scores from only your training set
X_train, X_test, y_train, y_test = train_test_split(features, labels)

fs = ReliefF()
fs.fit(X_train, y_train)

for feature_name, feature_score in zip(genetic_data.drop('class', axis=1).columns,
                                       fs.feature_importances_):
    print(feature_name, '\t', feature_score)

>>>N0 	 -0.0000166666666667
>>>N1 	 -0.006175
>>>N2 	 -0.0079
>>>N3 	 -0.006275
>>>N4 	 -0.00684166666667
>>>N5 	 -0.0104416666667
>>>N6 	 -0.010275
>>>N7 	 -0.00785
>>>N8 	 -0.00824166666667
>>>N9 	 -0.00515
>>>N10 	 -0.000216666666667
>>>N11 	 -0.0039
>>>N12 	 -0.00291666666667
>>>N13 	 -0.00345833333333
>>>N14 	 -0.00324166666667
>>>N15 	 -0.00886666666667
>>>N16 	 -0.00611666666667
>>>N17 	 -0.007325
>>>P1 	 0.108966666667
>>>P2 	 0.111
```

In this second example we show how to obtain scores when using ReliefF in combination with TuRF (some slight differences). In this example we differently assume that the loaded dataset is the training dataset and we do not need to split the data into training and testing prior to running ReliefF. The main difference here is that when using TuRF, fs.fit also requires 'headers' as an argument.  
```
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate.turf import TuRF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values
headers = list(genetic_data.drop("class", axis=1))
fs = TuRF(core_algorithm="ReliefF", n_features_to_select=2, pct=0.5,verbose=True)
fs.fit(features, labels, headers)
for feature_name, feature_score in zip(genetic_data.drop('class', axis=1).columns, fs.feature_importances_):
    print(feature_name, '\t', feature_score)
    
>>>N0 	 -0.00103125
>>>N1 	 -0.0107515625
>>>N2 	 -0.012890625
>>>N3 	 -0.012890625
>>>N4 	 -0.012890625
>>>N5 	 -0.012890625
>>>N6 	 -0.012890625
>>>N7 	 -0.012890625
>>>N8 	 -0.0107515625
>>>N9 	 -0.012890625
>>>N10 	 -0.00118125
>>>N11 	 -0.012890625
>>>N12 	 -0.0107515625
>>>N13 	 -0.0086125
>>>N14 	 -0.0107515625
>>>N15 	 -0.012890625
>>>N16 	 -0.0107515625
>>>N17 	 -0.012890625
>>>P1 	 0.20529375
>>>P2 	 0.17374375
```

To retrieve an np.array of feature importance scores in the original dataset ordering then add the following... (this also works with any core algorithm)
```
print(fs.feature\_importances\_)
>>>[-0.00103125 -0.01075156 -0.01289062 -0.01289062 -0.01289062 -0.01289062
 -0.01289062 -0.01289062 -0.01075156 -0.01289062 -0.00118125 -0.01289062
 -0.01075156 -0.0086125  -0.01075156 -0.01289062 -0.01075156 -0.01289062
  0.20529375  0.17374375]
```

To retrieve a list of indices for the top scoring features ranked by increasing score, then add the following... (this also works with any core algorithm)

```
print(fs.top\_features_)
>>>[13, 0, 10, 19, 18]
```

To sort features by decreasing score along with their names, and simultaneously indicate which features have been assigned a token TuRF feature score (since they were removed from consideration at some point) then add the following...

```
scored\_features = len(fs.top\_features_)
sorted_names = sorted(scoreDict, key=lambda x: scoreDict[x], reverse=True)
n = 1
for k in sorted\_names:
    if n < scored\_features +1 :
        print(k, '\t', scoreDict[k],'\t',n) 
    else:
        print(k, '\t', scoreDict[k],'\t','*') 
    n += 1
    
>>>P1 	 0.20529375 	 1
>>>P2 	 0.17374375 	 2
>>>N0 	 -0.00103125 	 3
>>>N10 	 -0.00118125 	 4
>>>N13 	 -0.0086125 	 5
>>>N1 	 -0.0107515625 	 *
>>>N14 	 -0.0107515625 	 *
>>>N16 	 -0.0107515625 	 *
>>>N8 	 -0.0107515625 	 *
>>>N12 	 -0.0107515625 	 *
>>>N3 	 -0.012890625 	 *
>>>N2 	 -0.012890625 	 *
>>>N7 	 -0.012890625 	 *
>>>N17 	 -0.012890625 	 *
>>>N5 	 -0.012890625 	 *
>>>N15 	 -0.012890625 	 *
>>>N11 	 -0.012890625 	 *
>>>N4 	 -0.012890625 	 *
>>>N9 	 -0.012890625 	 *
>>>N6 	 -0.012890625 	 *
```

Lastly, to output these scores to a text file in a format similar to how it is done in our alternative implementation of stand alone [ReBATE](https://github.com/EpistasisLab/ReBATE), add something like the following...

```
algorithm = 'TuRF_ReliefF'
discreteLimit = '10'
numNeighbors = '100'
outfile = algorithm + '-scores-' + discreteLimit + '-' + numNeighbors + '-' + 'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.txt'
fh = open(outfile, 'w')
fh.write(algorithm + ' Analysis Completed with REBATE\n')
fh.write('Run Time (sec): ' + str('NA') + '\n')
fh.write('=== SCORES ===\n')
n = 1
for k in sorted_names:
    if n < scored_features +1 :
        fh.write(str(k) + '\t' + str(scoreDict[k])  + '\t' + str(n) +'\n')
    else:
        fh.write(str(k) + '\t' + str(scoreDict[k])  + '\t' + '*' +'\n')
    n+=1
fh.close()
```

This ordered list and text output can be achieved similarly for any core Relief-based algorithm by just removing the 'if n < scored\_features +1 :' loop and the else statement adding the '*'. 




## General Usage Guidelines

1.) When performing feature selection, there is no universally best way to determine where to draw the cuttoff for including features. When using original Relief or ReliefF it has been suggested that features yielding a negative value score, can be confidently filtered out. This guideline is believed to be extendable to SURF, SURF\*, MultiSURF\*, and MultiSURF, however please note that features with a negative score are not necessarily irrelevant, and those with a positive score are not necessarily relevant. Instead, scores are most effectively interpreted as the relative evidence that a given feature is predictive of outcome. Thus, while it may be reasonable to only filter out features with a negative score, in practice it may be more useful to select some 'top' number of features to pass onto modeling. 

2.) In very large feature spaces users can expect core Relief-based algorithm scores to become less reliable when run on their own. This is because as the feature space becomes very large, the determination of nearest neighbors becomes more random.  As a result, in very large feature spaces (e.g. > 10,000 features), users should consider combining a core Relief-based algorithm with an iterative approach such as TuRF (implemented here) or VLSRelieF, or Iterative Relief. 

3.) When scaling up to big data problems, keep in mind that the data aspect that slows down ReBATE methods the most is the number of training instances, since Relief algorithms scale linearly with the number of features, but quadratically with the number of training instances. This is is the result of Relief-based methods needing to calculate a distance array (i.e. all pairwise distances between instances in the training dataset).  If you have a very large number of training instances available, consider utilizing a class balanced random sampling of that dataset when running any ReBATE methods to save on memory and computational time. 

