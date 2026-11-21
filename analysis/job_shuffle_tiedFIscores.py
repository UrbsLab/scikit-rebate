import argparse
import os
import pandas as pd
import numpy as np
import hashlib

def shuffle_ties(results_file, output_dir):
    # Reading Feature_Importance as a string to preserve the original values (i.e. the original decimal precision); no truncating
    df = pd.read_csv(results_file, sep='\t', dtype={'Feature_Importance': str})

    shuffled_groups = []

    with open(results_file, 'rb') as f: # open the file in read-binary mode
        file_hash = hashlib.sha256(f.read()).hexdigest() # read the entire file as bytes with f.read(), calculate a SHA-256 cryptographic hash of those bytes with hashlib.sha256(), return the hash as a hexadecimal string (64 hexadecimal characters) with .hexdigest()
    seed = int(file_hash[:16], 16) # generating seed from the content of the Results.txt file; extract the first 16 characters of the hexadecimal string, convert that value into a decimal integer
    rng = np.random.default_rng(seed) # reproducible, but ensures, for example, that every 2 row permutation is not the same (as it would be with random_state=int in .sample())

    # Iterate through groups with identical Feature_Importance values
    for importance, group in df.groupby('Feature_Importance', sort=False):

        # # Only shuffle groups containing multiple rows and at least one
        # # predictive feature (i.e. a Feature starting with 'M')
        # if len(group) > 1 and group['Feature'].str.startswith('M').any():
        #     group = group.sample(frac=1, random_state=42)
        # Only shuffle groups containing multiple rows
        if len(group) > 1:
            group = group.sample(frac=1, random_state=rng)

        shuffled_groups.append(group)

    # Recombine groups while preserving their original importance ordering
    shuffled_df = pd.concat(shuffled_groups, ignore_index=True)

    base_name = os.path.basename(results_file)
    results_file_name_without_txt = os.path.splitext(base_name)[0]
    # The path where shuffled_df will be saved
    output_path = os.path.join(output_dir, f"{results_file_name_without_txt}_shuffledTies.txt")
    
    # Save the shuffled DataFrame as a CSV
    shuffled_df.to_csv(output_path, index=False, sep='\t')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Path to the Results.txt file')
    args = parser.parse_args()

    # This will take "../Results/MultiSURF" from "../Results/MultiSURF/File01_Results.txt"
    alg_results_dir = os.path.dirname(args.input_file)
    # Create directory to store output Results files with ties handled (Results_shuffledTies.txt files)
    output_dir = os.path.join(alg_results_dir, "Results_shuffledTies")
    os.makedirs(output_dir, exist_ok=True)

    # Randomly shuffle ties involving predictive features in the input (Results.txt) file
    shuffle_ties(args.input_file, output_dir)

if __name__ == "__main__":
    main()