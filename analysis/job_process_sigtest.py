# job_process_sigtest.py
import os
import argparse
import pandas as pd
import numpy as np
from itertools import combinations
# from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon, binomtest, PermutationMethod
from statsmodels.stats.multitest import multipletests

def wilcoxon_signed_rank_test(rba_groups):
    results = []
    
    for rba1, rba2 in combinations(rba_groups.keys(), 2):
        x = rba_groups[rba1]
        y = rba_groups[rba2]

        d = x - y

        if np.all(d == 0): # handling the edge case where all differences between 2 algorithms are 0 (identical performance), avoiding a p-value being reported as NaN
            wilcoxon_statistic = 0.0
            wilcoxon_pvalue = 1.0
        else:
            # using permutation-based approach; presence of ties between algorithms (i.e. zero differences) can lead to inaccurate p-values with method="asymptotic" combined with a small sample size
            wilcoxon_res = wilcoxon(x, y, alternative="two-sided", method=PermutationMethod(n_resamples=10000, random_state=42), zero_method="wilcox")

            wilcoxon_statistic = wilcoxon_res.statistic
            wilcoxon_pvalue = wilcoxon_res.pvalue

        row = {
            'RBA1': rba1,
            'RBA2': rba2,
            'wilcoxon_statistic': wilcoxon_statistic,
            'wilcoxon_pvalue': wilcoxon_pvalue
        }
    
        results.append(row)
    
    results_df = pd.DataFrame(results)

    return results_df

def paired_sign_test(rba_groups):
    results = []
    
    for rba1, rba2 in combinations(rba_groups.keys(), 2):
        x = rba_groups[rba1]
        y = rba_groups[rba2]
    
        d = x - y
        d = d[d != 0] # removing zero differences (just like zero_method="wilcox" in scipy.stats.wilcoxon)

        if len(d) == 0: # handling the edge case where all differences between 2 algorithms are 0 (identical performance)
            proportion_positive_signs = np.nan
            paired_sign_pvalue = 1.0
        else:
            n_positive = np.sum(d > 0)
            n_negative = np.sum(d < 0)

            # performing paired sign test
            paired_sign_res = binomtest(
                n_positive,
                n_positive + n_negative,
                p=0.5,
                alternative='two-sided'
            )

            proportion_positive_signs = paired_sign_res.statistic
            paired_sign_pvalue = paired_sign_res.pvalue

        row = {
            'RBA1': rba1,
            'RBA2': rba2,
            'proportion_positive_signs': proportion_positive_signs,
            'paired_sign_pvalue': paired_sign_pvalue
        }

        results.append(row)

    results_df = pd.DataFrame(results)

    return results_df

def process_dir(dir_path, test, exclude_patterns=None):
    exclude_patterns = exclude_patterns or []
    df = pd.read_csv(os.path.join(dir_path, 'rankings_list.csv'), comment='#')

    # # Group by RBA
    # rba_groups = {rba: g['Rank'].values for rba, g in df.groupby('RBA') 
    #               if not any(p.lower() in rba.lower() for p in exclude_patterns)}
    # Group by RBA, calculating the mean predictive feature ranking per replicate as individual data points
    rba_groups = {
        rba: group.groupby('Dataset')['Rank'].mean().sort_index().values
        for rba, group in df.groupby('RBA')
        if not any(p.lower() in rba.lower() for p in exclude_patterns)
    }

    # Make sure RBAs all possess the same data
    datasets = {
        rba: set(group['Dataset'].unique())
        for rba, group in df.groupby('RBA')
        if not any(p.lower() in rba.lower() for p in exclude_patterns)
    }
    assert len(set(map(frozenset, datasets.values()))) == 1, \
        "RBAs do not contain the same replicate datasets."

    if test == "wilcoxon":
        results_df = wilcoxon_signed_rank_test(rba_groups)
    else: # paired sign test
        results_df = paired_sign_test(rba_groups)
    
    # results = []

    # for rba1, rba2 in combinations(rba_groups.keys(), 2):
    #     x = rba_groups[rba1]
    #     y = rba_groups[rba2]
        
    #     mannwhitney_res = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    #     row = {
    #         'RBA1': rba1,
    #         'RBA2': rba2,
    #         'mannwhitney_statistic': mannwhitney_res.statistic,
    #         'mannwhitney_pvalue': mannwhitney_res.pvalue
    #     }

    #     # Permutation test only for rank
    #     if column == 'rank':
    #         perm_p = permutation_test(x, y, mannwhitney_res.statistic)
    #         row['permutation_pvalue'] = perm_p

    #     results.append(row)

    # results_df = pd.DataFrame(results)

    # Sorting
    if test == "wilcoxon":
        # Benjamini-Hochberg
        results_df['wilcoxon_p_adj'] = multipletests(results_df['wilcoxon_pvalue'], method='fdr_bh')[1]
        results_df.sort_values(by='wilcoxon_pvalue', ascending=True, inplace=True)
        output_file = os.path.join(dir_path, 'wilcoxon_ranks.csv')
    else:
        # Benjamini-Hochberg
        results_df['paired_sign_p_adj'] = multipletests(results_df['paired_sign_pvalue'], method='fdr_bh')[1]
        results_df.sort_values(by='paired_sign_pvalue', ascending=True, inplace=True)
        output_file = os.path.join(dir_path, 'paired_sign_ranks.csv')

    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", help="Directory containing rankings_list.csv")
    parser.add_argument("--test", choices=['paired_sign', 'wilcoxon'], help="Test to conduct")
    parser.add_argument("--exclude", nargs='*', default=[], help="RBA name patterns to exclude (case insensitive)")
    args = parser.parse_args()

    process_dir(args.dir_path, test=args.test, exclude_patterns=args.exclude)