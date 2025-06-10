Assessing the Limitations of Relief-Based Algorithms in Detecting Higher-Order Interactions
Freda et al. 2024

README: Script descriptions

MasterRunScript.ipynb: This notebook will run Random Shuffle, Mutual Information, and all RBAs in a designated user directory. 
It stores all results, including both standard and absolute rankings for RBAs, in a new directory called 'Results'. 
Results will be stored within subdirectories entitled using the names of each method. 
This can be run globally but we suggest to run this script within each dataset group (patten of association - See Table 1 in the paper for further information).

Heatmaps.ipynb: This notebook consolidates all rankings, calculates the percentages of best rankings, and constructs heatmaps for visualization. 
This script will need to be run for each dataset configuration within each dataset group (patten of association).

MScoring.ipynb: This notebook calculates the average feature importances, derived from RBA methods and Mutual Information, for each predictive feature per experiment. 
It also produces a master sheet that consolidates all average feature importances for predictive features across all configurations within a dataset group (pattern of association) (e.g., XOR datasets).

NScoring.ipynb: This notebook does the same as above but for non-predictive features. 
Instead of outputting the average scores per predictive feature, this script averages across all non-predictive features within and across experiments. 
As with Mscoring.ipynb, this will need to be run for each dataset group (pattern of association)

Data_feature_set_extender.ipynb: This notebook adds non-predictive features to existing datasets (i.e. from 20 to 100 features) for this study using datasets taken from the original Relief benchmarking paper (Urbanowicz 2018).
