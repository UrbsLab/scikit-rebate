import os
import sys
import argparse
import pandas as pd
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import hashlib

package_path = os.path.abspath(os.path.join("", ".."))
sys.path.insert(0, package_path)
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, SWRFstar, SWRFstar2, SWRF, TBD1star, TBD1, TBD2star, TBD2, TBD2linearstar, TBD2linear, TBD2exponentialstar, TBD2exponential, TBD2linear3SDstar, TBD2linear3SD, TBD2exponential3SDstar, TBD2exponential3SD, MuRelief

# NEW: added exist_ok=True
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def process_and_save_results(file_path, fs, method_name):
    df = pd.read_csv(file_path, sep='\t')
    # features, labels = df.drop('Class', axis=1).values, df['Class'].values
    X, y = df.drop('Class', axis=1).values, df['Class'].values
    # X_train, _, y_train, _ = train_test_split(features, labels)

    try:
        fs.fit(X, y)
    except AttributeError:
        scores = fs(X, y)
        fs = type('MI', (), {'feature_importances_': scores})()

    temp_list = []
    for feature_name, feature_score in zip(df.drop('Class', axis=1).columns, fs.feature_importances_):
        temp_list.append([feature_name, feature_score])
    
    Results = pd.DataFrame(temp_list, columns=['Feature', 'Feature_Importance'])
    ABSResults = Results.copy()
    ABSResults['ABS_Feature_Importance'] = ABSResults['Feature_Importance'].abs()
    Results.sort_values(by='Feature_Importance', ascending=False, inplace=True)
    ABSResults.sort_values(by='ABS_Feature_Importance', ascending=False, inplace=True)

    base_dir = os.path.dirname(file_path)
    results_dir = os.path.join(base_dir, "Results")
    method_dir = os.path.join(results_dir, method_name)
    abs_method_dir = os.path.join(results_dir, f"ABS_{method_name}")
    ensure_dir(method_dir)
    ensure_dir(abs_method_dir)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    Results.to_csv(os.path.join(method_dir, f"{base_name}_Results.txt"), index=False, sep='\t')
    ABSResults.to_csv(os.path.join(abs_method_dir, f"{base_name}_ABSResults.txt"), index=False, sep='\t')

    if method_name in ["SWRFstar2", "SWRF", "TBD1", "TBD1star", "TBD2", "TBD2star", "TBD2linear", "TBD2linearstar", "TBD2exponential", "TBD2exponentialstar", "TBD2linear3SD", "TBD2linear3SDstar", "TBD2exponential3SD", "TBD2exponential3SDstar", "SURF", "SURFstar", "MultiSURF", "MultiSURFstar", "MuRelief"]:
        # fs.plot_distance_weight_map(save_fig=os.path.join(method_dir, f"{base_name}_WeightPlot.png"), show_expected=True)
        fs.plot_distance_weight_map(save_fig=os.path.join(method_dir, f"{base_name}_WeightPlot.png"), save_fig=os.path.join(method_dir, f"{base_name}_stdweightlog.txt"), show_expected=True)

    print(f"Processed {file_path} with {method_name}. Results saved to {method_dir} and {abs_method_dir}.")

def process_random_shuffle(file_path):
    # Define the directory to store results
    results_dir = os.path.join(os.path.dirname(file_path), "Results", "RandomShuffle")
    ensure_dir(results_dir)

    # Read the file
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Shuffle feature column names (excluding 'Class' if present)
    if 'Class' in df.columns:
        columns_to_shuffle = df.drop('Class', axis=1).columns.tolist()
    else:
        columns_to_shuffle = df.columns.tolist()

    # shuffled_columns = np.random.permutation(columns_to_shuffle)
    # NEW: reproducible shuffling based on file name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # creating a deterministic seed based on the file name
    seed = int(hashlib.sha256(base_name.encode()).hexdigest(), 16) % (2**32)
    # creating a local RNG seeded from the file name
    rng = default_rng(seed)
    # shuffle columns deterministically for this file
    shuffled_columns = rng.permutation(columns_to_shuffle)

    shuffled_df = pd.DataFrame(shuffled_columns, columns=['Feature'])

    # base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(results_dir, f"{base_name}_RandShuffle.txt")
    shuffled_df.to_csv(output_path, index=False, sep='\t')

def process_mutual_info(file_path):
    # df = pd.read_csv(file_path, sep='\t')
    # features, labels = df.drop('Class', axis=1).values, df['Class'].values
    # X_train, _, y_train, _ = train_test_split(features, labels)
    # scores = mutual_info_classif(X_train, y_train)
    # fs = type('MI', (), {'feature_importances_': scores})()  # Mock object with same interface
    fs = mutual_info_classif
    process_and_save_results(file_path, fs, "MutualInfo")

def process_relieff10(file_path):
    fs = ReliefF(n_features_to_select=2,n_neighbors=10)
    process_and_save_results(file_path, fs, "ReliefF10")

def process_relieff100(file_path):
    fs = ReliefF(n_features_to_select=2,n_neighbors=100)
    process_and_save_results(file_path, fs, "ReliefF100")

def process_surf(file_path):
    fs = SURF()
    process_and_save_results(file_path, fs, "SURF")

def process_surfstar(file_path):
    fs = SURFstar()
    process_and_save_results(file_path, fs, "SURFstar")

def process_multisurf(file_path):
    fs = MultiSURF()
    process_and_save_results(file_path, fs, "MultiSURF")

def process_multisurfstar(file_path):
    fs = MultiSURFstar()
    process_and_save_results(file_path, fs, "MultiSURFstar")

def process_swrfstar(file_path):
    fs = SWRFstar()
    process_and_save_results(file_path, fs, "SWRFstar")

def process_swrfstar2(file_path):
    fs = SWRFstar2()
    process_and_save_results(file_path, fs, "SWRFstar2")

def process_swrf(file_path):
    fs = SWRF()
    process_and_save_results(file_path, fs, "SWRF")

def process_TBD1star(file_path):
    fs = TBD1star()
    process_and_save_results(file_path, fs, "TBD1star")

def process_TBD1(file_path):
    fs = TBD1()
    process_and_save_results(file_path, fs, "TBD1")

def process_TBD2star(file_path):
    fs = TBD2star()
    process_and_save_results(file_path, fs, "TBD2star")

def process_TBD2(file_path):
    fs = TBD2()
    process_and_save_results(file_path, fs, "TBD2")

def process_TBD2linearstar(file_path):
    fs = TBD2linearstar()
    process_and_save_results(file_path, fs, "TBD2linearstar")

def process_TBD2linear(file_path):
    fs = TBD2linear()
    process_and_save_results(file_path, fs, "TBD2linear")

def process_TBD2exponentialstar(file_path):
    fs = TBD2exponentialstar()
    process_and_save_results(file_path, fs, "TBD2exponentialstar")

def process_TBD2exponential(file_path):
    fs = TBD2exponential()
    process_and_save_results(file_path, fs, "TBD2exponential")

# 3 SD versions of TBD2 variants:
def process_TBD2linear3SDstar(file_path):
    fs = TBD2linear3SDstar()
    process_and_save_results(file_path, fs, "TBD2linear3SDstar")

def process_TBD2linear3SD(file_path):
    fs = TBD2linear3SD()
    process_and_save_results(file_path, fs, "TBD2linear3SD")

def process_TBD2exponential3SDstar(file_path):
    fs = TBD2exponential3SDstar()
    process_and_save_results(file_path, fs, "TBD2exponential3SDstar")

def process_TBD2exponential3SD(file_path):
    fs = TBD2exponential3SD()
    process_and_save_results(file_path, fs, "TBD2exponential3SD")

def process_murelief(file_path):
    fs = MuRelief()
    process_and_save_results(file_path, fs, "MuRelief")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, help='Algorithm to use')
    parser.add_argument('--input_file', required=True, help='Path to the input .txt file')
    args = parser.parse_args()

    alg_map = {
        'random': process_random_shuffle,
        'mutual_info': process_mutual_info,
        'relieff10': process_relieff10,
        'relieff100': process_relieff100,
        'surf': process_surf,
        'surfstar': process_surfstar,
        'multisurf': process_multisurf,
        'multisurfstar': process_multisurfstar,
        'swrfstar': process_swrfstar,
        'swrfstar2': process_swrfstar2,
        'swrf': process_swrf,
        'tbd1star': process_TBD1star,
        'tbd1': process_TBD1,
        'tbd2star': process_TBD2star,
        'tbd2': process_TBD2,
        'tbd2linearstar': process_TBD2linearstar,
        'tbd2linear': process_TBD2linear,
        'tbd2exponentialstar': process_TBD2exponentialstar,
        'tbd2exponential': process_TBD2exponential,
        'tbd2linear3sdstar': process_TBD2linear3SDstar,
        'tbd2linear3sd': process_TBD2linear3SD,
        'tbd2exponential3sdstar': process_TBD2exponential3SDstar,
        'tbd2exponential3sd': process_TBD2exponential3SD,
        'murelief': process_murelief,
    }

    if args.algorithm not in alg_map:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    
    alg_map[args.algorithm](args.input_file)

if __name__ == "__main__":
    main()
