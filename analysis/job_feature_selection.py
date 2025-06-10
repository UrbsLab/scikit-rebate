import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, SWRFstar
from sklearn.feature_selection import mutual_info_classif

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_mutual_info(df, output_path):
    features, labels = df.drop('Class', axis=1).values, df['Class'].values
    X_train, _, y_train, _ = train_test_split(features, labels)
    mi_scores = mutual_info_classif(X_train, y_train)
    result = pd.DataFrame({'Feature': df.drop('Class', axis=1).columns, 'Feature_Importance': mi_scores})
    result.sort_values(by='Feature_Importance', ascending=False, inplace=True)
    result.to_csv(output_path, index=False, sep='\\t')

def process_relieff(df, output_path, k):
    features, labels = df.drop('Class', axis=1).values, df['Class'].values
    X_train, _, y_train, _ = train_test_split(features, labels)
    fs = ReliefF(n_features_to_select=2, n_neighbors=k)
    fs.fit(X_train, y_train)
    result = pd.DataFrame({'Feature': df.drop('Class', axis=1).columns, 'Feature_Importance': fs.feature_importances_})
    result.sort_values(by='Feature_Importance', ascending=False, inplace=True)
    result.to_csv(output_path, index=False, sep='\\t')

def process_surf(df, output_path):
    features, labels = df.drop('Class', axis=1).values, df['Class'].values
    X_train, _, y_train, _ = train_test_split(features, labels)
    fs = SURF()
    fs.fit(X_train, y_train)
    result = pd.DataFrame({'Feature': df.drop('Class', axis=1).columns, 'Feature_Importance': fs.feature_importances_})
    result.sort_values(by='Feature_Importance', ascending=False, inplace=True)
    result.to_csv(output_path, index=False, sep='\\t')

def process_surfstar(df, output_path):
    features, labels = df.drop('Class', axis=1).values, df['Class'].values
    X_train, _, y_train, _ = train_test_split(features, labels)
    fs = SURFstar()
    fs.fit(X_train, y_train)
    result = pd.DataFrame({'Feature': df.drop('Class', axis=1).columns, 'Feature_Importance': fs.feature_importances_})
    result.sort_values(by='Feature_Importance', ascending=False, inplace=True)
    result.to_csv(output_path, index=False, sep='\\t')

def process_multisurf(df, output_path):
    features, labels = df.drop('Class', axis=1).values, df['Class'].values
    X_train, _, y_train, _ = train_test_split(features, labels)
    fs = MultiSURF()
    fs.fit(X_train, y_train)
    result = pd.DataFrame({'Feature': df.drop('Class', axis=1).columns, 'Feature_Importance': fs.feature_importances_})
    result.sort_values(by='Feature_Importance', ascending=False, inplace=True)
    result.to_csv(output_path, index=False, sep='\\t')

def process_multisurfstar(df, output_path):
    features, labels = df.drop('Class', axis=1).values, df['Class'].values
    X_train, _, y_train, _ = train_test_split(features, labels)
    fs = MultiSURFstar()
    fs.fit(X_train, y_train)
    result = pd.DataFrame({'Feature': df.drop('Class', axis=1).columns, 'Feature_Importance': fs.feature_importances_})
    result.sort_values(by='Feature_Importance', ascending=False, inplace=True)
    result.to_csv(output_path, index=False, sep='\\t')

def process_swrfstar(df, output_path):
    features, labels = df.drop('Class', axis=1).values, df['Class'].values
    X_train, _, y_train, _ = train_test_split(features, labels)
    fs = SWRFstar()
    fs.fit(X_train, y_train)
    result = pd.DataFrame({'Feature': df.drop('Class', axis=1).columns, 'Feature_Importance': fs.feature_importances_})
    result.sort_values(by='Feature_Importance', ascending=False, inplace=True)
    result.to_csv(output_path, index=False, sep='\\t')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, help='Algorithm to use: mutual_info, relieff10, relieff100, multisurf, multisurfstar')
    parser.add_argument('--input_file', required=True, help='Path to the input .txt file')
    parser.add_argument('--output_dir', required=True, help='Directory to save results')
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, sep='\\t')
    ensure_dir(args.output_dir)
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_file = os.path.join(args.output_dir, f"{base_name}_{args.algorithm}_results.txt")

    if args.algorithm == 'mutual_info':
        process_mutual_info(df, output_file)
    elif args.algorithm == 'relieff10':
        process_relieff(df, output_file, k=10)
    elif args.algorithm == 'multisurf':
        process_multisurf(df, output_file)
    elif args.algorithm == 'multisurfstar':
        process_multisurfstar(df, output_file)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

if __name__ == "__main__":
    main()