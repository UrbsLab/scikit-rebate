import os
import time
import sys
import argparse
import pandas as pd
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import hashlib

package_path = os.path.abspath(os.path.join("", ".."))
sys.path.insert(0, package_path)
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, SWRFstar, SWRFstar2, SWRF, MultiSWRFstar, MultiSWRF, MultiSWRFDBstar, MultiSWRFDB, MultiSWRFDBlinearstar, MultiSWRFDBlinear, MultiSWRFDBexponentialstar, MultiSWRFDBexponential, MultiSWRFDBlinear3SDstar, MultiSWRFDBlinear3SD, MultiSWRFDBexponential3SDstar, MultiSWRFDBexponential3SD, MuRelief

# NEW: added exist_ok=True
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def process_and_save_results(file_path, fs, method_name):
    df = pd.read_csv(file_path, sep='\t')
    # features, labels = df.drop('Class', axis=1).values, df['Class'].values
    X, y = df.drop('Class', axis=1).values, df['Class'].values
    # X_train, _, y_train, _ = train_test_split(features, labels)

    # to keep track of runtime for large feature datasets
    start_time = time.time()
    try:
        fs.fit(X, y)
    except AttributeError:
        scores = fs(X, y)
        fs = type('MI', (), {'feature_importances_': scores})()
    end_time = time.time()

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
    # Save runtime CSV
    runtime = end_time - start_time
    runtime_df = pd.DataFrame([{
        "Dataset": base_name,
        "Algorithm": method_name,
        "Runtime (sec)": round(runtime, 4),
        "Runtime (min)": round(runtime / 60, 4)
    }])
    runtime_df.to_csv(os.path.join(method_dir, f"{base_name}_runtime_diagnosticrun.csv"), index=False)

    # ******** If I uncomment this function, I need to go back and uncomment logging lines within other files
    if method_name in ["SWRFstar2", "SWRF", "MultiSWRF", "MultiSWRFstar", "MultiSWRFDB", "MultiSWRFDBstar", "MultiSWRFDBlinear", "MultiSWRFDBlinearstar", "MultiSWRFDBexponential", "MultiSWRFDBexponentialstar", "MultiSWRFDBlinear3SD", "MultiSWRFDBlinear3SDstar", "MultiSWRFDBexponential3SD", "MultiSWRFDBexponential3SDstar", "SURF", "SURFstar", "MultiSURF", "MultiSURFstar", "MuRelief10", "MuRelief100"]:
        fs.plot_distance_weight_map(save_fig=os.path.join(method_dir, f"{base_name}_WeightPlot.png"), show_expected=True)
        # fs.plot_distance_weight_map(save_fig=os.path.join(method_dir, f"{base_name}_WeightPlot.png"), save_file=os.path.join(method_dir, f"{base_name}_stdweightlog.txt"), show_expected=True)

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
    df = pd.read_csv(file_path, sep='\t')
    # counting the number of labels in the 'Class' column to determine whether this is a classification or regression problem:
    num_labels = df['Class'].nunique()
    if num_labels <= 10:
        fs = mutual_info_classif
    else:
        fs = mutual_info_regression
    process_and_save_results(file_path, fs, "MutualInfo")

def process_relieff10(file_path):
    fs = ReliefF(n_features_to_select=2,n_neighbors=10,n_jobs=16)
    process_and_save_results(file_path, fs, "ReliefF10")

def process_relieff100(file_path):
    fs = ReliefF(n_features_to_select=2,n_neighbors=100,n_jobs=16)
    process_and_save_results(file_path, fs, "ReliefF100")

def process_surf(file_path):
    fs = SURF(n_jobs=16)
    process_and_save_results(file_path, fs, "SURF")

def process_surfstar(file_path):
    fs = SURFstar(n_jobs=16)
    process_and_save_results(file_path, fs, "SURFstar")

def process_multisurf(file_path):
    fs = MultiSURF(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSURF")

def process_multisurfstar(file_path):
    fs = MultiSURFstar(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSURFstar")

def process_swrfstar(file_path):
    fs = SWRFstar(n_jobs=1)
    process_and_save_results(file_path, fs, "SWRFstar")

def process_swrfstar2(file_path):
    fs = SWRFstar2(n_jobs=1)
    process_and_save_results(file_path, fs, "SWRFstar2")

def process_swrf(file_path):
    fs = SWRF(n_jobs=1)
    process_and_save_results(file_path, fs, "SWRF")

def process_multiswrfstar(file_path):
    fs = MultiSWRFstar(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFstar")

def process_multiswrf(file_path):
    fs = MultiSWRF(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRF")

def process_multiswrfdbstar(file_path):
    fs = MultiSWRFDBstar(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFDBstar")

def process_multiswrfdb(file_path):
    fs = MultiSWRFDB(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFDB")

def process_multiswrfdblinearstar(file_path):
    fs = MultiSWRFDBlinearstar(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFDBlinearstar")

def process_multiswrfdblinear(file_path):
    fs = MultiSWRFDBlinear(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFDBlinear")

def process_multiswrfdbexponentialstar(file_path):
    fs = MultiSWRFDBexponentialstar(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFDBexponentialstar")

def process_multiswrfdbexponential(file_path):
    fs = MultiSWRFDBexponential(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFDBexponential")

# 3 SD versions of MultiSWRFDB variants:
def process_multiswrfdblinear3SDstar(file_path):
    fs = MultiSWRFDBlinear3SDstar(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFDBlinear3SDstar")

def process_multiswrfdblinear3SD(file_path):
    fs = MultiSWRFDBlinear3SD(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFDBlinear3SD")

def process_multiswrfdbexponential3SDstar(file_path):
    fs = MultiSWRFDBexponential3SDstar(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFDBexponential3SDstar")

def process_multiswrfdbexponential3SD(file_path):
    fs = MultiSWRFDBexponential3SD(n_jobs=1)
    process_and_save_results(file_path, fs, "MultiSWRFDBexponential3SD")

def process_murelief10(file_path):
    fs = MuRelief(n_features_to_select=2,n_neighbors=10,n_jobs=16)
    process_and_save_results(file_path, fs, "MuRelief10")

def process_murelief100(file_path):
    fs = MuRelief(n_features_to_select=2,n_neighbors=100,n_jobs=16)
    process_and_save_results(file_path, fs, "MuRelief100")


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
        'multiswrfstar': process_multiswrfstar,
        'multiswrf': process_multiswrf,
        'multiswrfdbstar': process_multiswrfdbstar,
        'multiswrfdb': process_multiswrfdb,
        'multiswrfdblinearstar': process_multiswrfdblinearstar,
        'multiswrfdblinear': process_multiswrfdblinear,
        'multiswrfdbexponentialstar': process_multiswrfdbexponentialstar,
        'multiswrfdbexponential': process_multiswrfdbexponential,
        'multiswrfdblinear3sdstar': process_multiswrfdblinear3SDstar,
        'multiswrfdblinear3sd': process_multiswrfdblinear3SD,
        'multiswrfdbexponential3sdstar': process_multiswrfdbexponential3SDstar,
        'multiswrfdbexponential3sd': process_multiswrfdbexponential3SD,
        'murelief10': process_murelief10,
        'murelief100': process_murelief100,
    }

    if args.algorithm not in alg_map:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    
    alg_map[args.algorithm](args.input_file)

if __name__ == "__main__":
    main()
