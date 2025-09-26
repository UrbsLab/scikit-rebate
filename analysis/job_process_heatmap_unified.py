import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

# Common helper to create the percentages_df (copied from your old job_process_heatmap)
def compute_percentages(results_dir):
    all_rankings_df = pd.DataFrame()
    total_features_per_rba = {}

    for rba in os.listdir(results_dir):
        rba_path = os.path.join(results_dir, rba)
        if os.path.isdir(rba_path):
            method_feature_counts = []
            for file in os.listdir(rba_path):
                if file.endswith('.txt'):
                    file_path = os.path.join(rba_path, file)
                    parts = file.split('_')
                    dataset_id = '_'.join(parts[-2].split('_')[:2])

                    if rba == "RandomShuffle":
                        df = pd.read_csv(file_path, sep='\t', usecols=['Feature'])
                        df['Rank'] = df.index + 1
                    else:
                        column_to_use = 'ABS_Feature_Importance' if "ABS" in rba else 'Feature_Importance'
                        df = pd.read_csv(file_path, sep='\t', usecols=['Feature', column_to_use])
                        df.sort_values(by=column_to_use, ascending=False, inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        df['Rank'] = df.index + 1

                    predictive_df = df[df['Feature'].str.startswith('M')][['Feature', 'Rank']]
                    predictive_df['RBA'] = rba
                    predictive_df['Dataset'] = dataset_id

                    all_rankings_df = pd.concat([all_rankings_df, predictive_df], ignore_index=True)
                    method_feature_counts.append(df['Feature'].nunique())

            total_features_per_rba[rba] = max(method_feature_counts) if method_feature_counts else 0

    N = next(iter(total_features_per_rba.values()))
    lowest_ranks = all_rankings_df.groupby(['RBA', 'Dataset'])['Rank'].max().reset_index()
    percentages = {rba: [0] * N for rba in lowest_ranks['RBA'].unique()}
    for rba in percentages:
        rba_data = lowest_ranks[lowest_ranks['RBA'] == rba]
        for pos in range(1, N + 1):
            count_higher = rba_data[rba_data['Rank'] <= pos].shape[0]
            percentages[rba][pos - 1] = (count_higher / rba_data.shape[0]) * 100 if rba_data.shape[0] > 0 else 0

    percentages_df = pd.DataFrame(percentages, index=range(1, N + 1))
    n_pred = all_rankings_df['Feature'].nunique()
    return percentages_df.iloc[n_pred-1:]  # same trimming

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("basedir", help="Directory containing multiple Results folders.")
    parser.add_argument("--prefix", default="", help="Prefix for unified PDF filename.")
    args = parser.parse_args()

    # custom colormap & ordering
    custom_cmap = sns.color_palette('Oranges', n_colors=1000)[:800] + sns.color_palette('Blues', n_colors=1000)[800:]

    rba_order = [
        'RandomShuffle','MutualInfo','ReliefF10','ReliefF100','SURF','SURFstar',
        'MultiSURF','MultiSURFstar','SWRF','SWRFstar2','TBD1','TBD1star','TBD2','TBD2star'
    ]

    # Grab all Results folders
    results_dirs = []
    for root, dirs, _ in os.walk(args.basedir):
        for d in dirs:
            if d == 'Results':
                results_dirs.append(os.path.join(root, d))

    # Build a mapping of (n_instances, heritability, EDMtype) -> percentages_df
    pattern_n = re.compile(r"s_(\d+)")
    pattern_h = re.compile(r"her_(\d+\.\d+)__maf")
    pattern_edm = re.compile(r"EDM-(\d+)")

    data_dict = {}
    for rd in results_dirs:
        n_match = pattern_n.search(rd)
        h_match = pattern_h.search(rd)
        edm_match = pattern_edm.search(rd)
        if n_match and h_match and edm_match:
            n = int(n_match.group(1))
            h = float(h_match.group(1))
            edm = edm_match.group(1)  # '1' or '2'
            perc_df = compute_percentages(rd)
            perc_df_T = perc_df.T
            perc_df_ordered = perc_df_T.loc[[r for r in rba_order if r in perc_df_T.index]]
            data_dict[(n, h, edm)] = perc_df_ordered

    n_values = sorted({k[0] for k in data_dict.keys()})
    h_values = sorted({k[1] for k in data_dict.keys()})

    fig, axes = plt.subplots(len(h_values)*2, len(n_values), figsize=(4*len(n_values), 3*len(h_values)*2))

    if len(h_values)*2 == 1 and len(n_values) == 1:
        axes = np.array([[axes]])  # ensure 2D

    xtick_labels = ['Optimal','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
    for i, h in enumerate(h_values):
        for j, n in enumerate(n_values):
            # top EDM-2
            ax_top = axes[i*2, j] if len(h_values)*2 > 1 else axes[0, j]
            ax_bot = axes[i*2+1, j] if len(h_values)*2 > 1 else axes[1, j]

            df2 = data_dict.get((n,h,'2'))
            df1 = data_dict.get((n,h,'1'))
            if df2 is not None:
                sns.heatmap(df2, ax=ax_top, annot=False, cmap=custom_cmap, cbar=False, xticklabels=False, yticklabels=False)
                # ax_top.set_title(f"n={n}, h={h}", fontsize=10)
            else:
                ax_top.axis('off')
            if df1 is not None:
                sns.heatmap(df1, ax=ax_bot, annot=False, cmap=custom_cmap, cbar=False, xticklabels=xtick_labels, yticklabels=False)
            else:
                ax_bot.axis('off')

            # if j==0:
            #     ax_top.set_ylabel("E", rotation=0, labelpad=20, fontsize=12)
            #     ax_bot.set_ylabel("H", rotation=0, labelpad=20, fontsize=12)
            # Put "E" and "H" labels on the right side of the heatmaps in the last column
            if j == len(n_values) - 1:
                ax_top.set_ylabel("E", rotation=0, labelpad=20, fontsize=12)
                ax_top.yaxis.set_label_position("right")
                
                ax_bot.set_ylabel("H", rotation=0, labelpad=20, fontsize=12)
                ax_bot.yaxis.set_label_position("right")

    # # Add one colorbar on the far right
    # cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    # # take any df to draw dummy heatmap for cbar
    # any_df = next(iter(data_dict.values()))
    # sns.heatmap(any_df, cmap=custom_cmap, cbar_ax=cbar_ax, cbar_kws={'label': 'Power (Frequency of Success)'}, annot=False)

    # fig.suptitle("Unified Heatmaps", fontsize=16)
    fig.supxlabel("Number of Training Instances (n)", fontsize=14)
    fig.supylabel("Heritability of Model", fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    outdir = os.path.basename(os.path.normpath(args.basedir))
    parentdir = os.path.dirname(os.path.normpath(args.basedir))
    save_path = os.path.join(parentdir, outdir, args.prefix + "unified_heatmaps.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Unified heatmap saved to {save_path}")

if __name__ == "__main__":
    main()