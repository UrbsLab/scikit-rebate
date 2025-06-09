import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, TURF

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_feature_selection(file_path, algorithm, output_subdir, algo_params=None, use_abs=True):
    genetic_data = pd.read_csv(file_path, sep='\t')
    features, labels = genetic_data.drop('Class', axis=1).values, genetic_data['Class'].values
    X_train, _, y_train, _ = train_test_split(features, labels)

    # Initialize and fit algorithm
    fs = algorithm(**(algo_params or {}))
    fs.fit(X_train, y_train)

    feature_names = genetic_data.drop('Class', axis=1).columns
    scores = fs.feature_importances_

    results = pd.DataFrame({'Feature': feature_names, 'Feature_Importance': scores})
    results.sort_values(by='Feature_Importance', ascending=False, inplace=True)

    base_dir = os.path.dirname(file_path)
    results_dir = os.path.join(base_dir, "Results", output_subdir)
    ensure_dir(results_dir)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    results.to_csv(os.path.join(results_dir, f"{base_name}_Results.txt"), index=False, sep='\t')

    if use_abs:
        abs_dir = os.path.join(base_dir, "Results", f"ABS_{output_subdir}")
        ensure_dir(abs_dir)
        abs_results = results.copy()
        abs_results['ABS_Feature_Importance'] = abs_results['Feature_Importance'].abs()
        abs_results.sort_values(by='ABS_Feature_Importance', ascending=False, inplace=True)
        abs_results.to_csv(os.path.join(abs_dir, f"{base_name}_ABSResults.txt"), index=False, sep='\t')

def find_and_run_algorithm(root_dir, algorithm, output_subdir, algo_params=None, use_abs=True):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "EDM" in os.path.basename(dirpath):
            for filename in filenames:
                if filename.endswith('.txt'):
                    file_path = os.path.join(dirpath, filename)
                    process_feature_selection(file_path, algorithm, output_subdir, algo_params, use_abs)
