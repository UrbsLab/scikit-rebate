import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from skrebate import ReliefF, MultiSURF, MultiSURFstar

ALGORITHMS = {
    "ReliefF": ReliefF,
    "MultiSURF": MultiSURF,
    "MultiSURFstar": MultiSURFstar
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_feature_selection(file_path, algo_name, output_subdir, n_neighbors=100, n_features_to_select=2):
    data = pd.read_csv(file_path, sep='\t')
    X, y = data.drop('Class', axis=1).values, data['Class'].values
    X_train, _, y_train, _ = train_test_split(X, y)

    algorithm = ALGORITHMS[algo_name](n_neighbors=n_neighbors, n_features_to_select=n_features_to_select) \
        if algo_name == "ReliefF" else ALGORITHMS[algo_name]()

    algorithm.fit(X_train, y_train)
    feature_names = data.drop('Class', axis=1).columns
    scores = algorithm.feature_importances_

    results = pd.DataFrame({'Feature': feature_names, 'Feature_Importance': scores})
    results.sort_values(by='Feature_Importance', ascending=False, inplace=True)

    base_dir = os.path.dirname(file_path)
    result_dir = os.path.join(base_dir, "Results", output_subdir)
    ensure_dir(result_dir)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    results.to_csv(os.path.join(result_dir, f"{base_name}_Results.txt"), sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--algorithm', type=str, choices=ALGORITHMS.keys(), required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--neighbors', type=int, default=100)
    parser.add_argument('--nfeat', type=int, default=2)
    args = parser.parse_args()

    process_feature_selection(args.file, args.algorithm, args.outdir, args.neighbors, args.nfeat)
