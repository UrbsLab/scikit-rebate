import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

# Helper to create the percentages_df (copied from old job_process_heatmap)
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
        'MultiSURF','MultiSURFstar','SWRF','SWRFstar2','MultiSWRF','MultiSWRFstar','MultiSWRFDB','MultiSWRFDBstar',
        'MuRelief10','MuRelief100'
    ]
    # rba_order = [
    #     'RandomShuffle', 'MutualInfo', 'MultiSWRFDB', 'MultiSWRFDBstar', 'MultiSWRFDBlinear', 'MultiSWRFDBlinear3SD', 'MultiSWRFDBlinearstar', 'MultiSWRFDBlinear3SDstar', 'MultiSWRFDBexponential', 'MultiSWRFDBexponential3SD', 'MultiSWRFDBexponentialstar', 'MultiSWRFDBexponential3SDstar'
    # ]
    # rba_order = [
    #     'MultiSWRFDB', 'MultiSWRFDBstar'
    # ]

    # Grab all Results folders (for 100 feature datasets)
    # * can later potentially add the dataset feature lengths you want for the heatmap as a parameter (ex. 100 for a_100 datasets)
    results_dirs = []
    for root, dirs, _ in os.walk(args.basedir):
        for d in dirs:
            if d == 'Results' and 'a_100' in root:
                results_dirs.append(os.path.join(root, d))

    is_xor = ('XOR' in args.basedir)
    is_mainEff_or_core2wayEpistasis = ('mainEff_Datasets' in args.basedir or 'core2wayEpistasis' in args.basedir)
    is_mainEffadditive_or_2wayEpiHet = ('mainEff_additive' in args.basedir or '2wayEpiHeterogeneity' in args.basedir)
    print("Is mainEffadditive or 2wayEpiHet:", is_mainEffadditive_or_2wayEpiHet)

    # Build a mapping of (n_instances, heritability, EDMtype) -> percentages_df
    # pattern_n = re.compile(r"s_(\d+)")
    pattern_n = re.compile(r"/s_(\d+)")
    if is_xor:
        pattern_h = re.compile(r"xor_(\d+)")
    elif is_mainEffadditive_or_2wayEpiHet:
        # pattern_h = re.compile(r"r_(\d+)")
        pattern_h = re.compile(r"/r_(\d+)")
    else:
        pattern_h = re.compile(r"her_(\d+\.\d+)__maf")
    pattern_edm = re.compile(r"EDM-(\d+)")

    data_dict = {}
    for rd in results_dirs:
        print("Results directory:", rd)
        n_match = pattern_n.search(rd)
        h_match = pattern_h.search(rd)
        edm_match = pattern_edm.search(rd)
        if n_match and h_match and edm_match:
            n = int(n_match.group(1))
            if is_xor:
                h = int(h_match.group(1))
            elif is_mainEffadditive_or_2wayEpiHet:
                # to turn r_75 or r_50 into 0.75 or 0.5
                h = int(h_match.group(1)) / 100
                print("H value:", h)
            else:
                h = float(h_match.group(1))
            edm = edm_match.group(1)  # '1' or '2'
            perc_df = compute_percentages(rd)
            perc_df_T = perc_df.T
            perc_df_ordered = perc_df_T.loc[[r for r in rba_order if r in perc_df_T.index]]
            data_dict[(n, h, edm)] = perc_df_ordered

    n_values = sorted({k[0] for k in data_dict.keys()})
    h_values = sorted({k[1] for k in data_dict.keys()})
    print("Data dictionary:", data_dict, "\n")
    print("H values:", h_values, "\n")
    print("N values:", n_values, "\n")

    # fig, axes = plt.subplots(len(h_values)*2, len(n_values), figsize=(4*len(n_values), 3*len(h_values)*2))
    # --- NEW ATTEMPT TO INCREASE SPACING
    # only mainEff and core2wayEpistasis have both EDM-1 and EDM-2
    if is_mainEff_or_core2wayEpistasis:
        total_rows = len(h_values) * 2
    else:
        total_rows = len(h_values)
    total_cols = len(n_values)

    # Create custom height and width ratios
    # --> every 2 rows, insert an extra gap by slightly enlarging the space below (for mainEff and core2wayEpistasis)
    # for all other dataset configurations, add the gap after every row
    height_ratios = []
    for i in range(total_rows):
        height_ratios.append(1)
        if is_mainEff_or_core2wayEpistasis:
            if i % 2 == 1 and i != total_rows - 1:  # after every 2nd row (except last)
                height_ratios.append(0.15)  # this adds vertical gap spacing
        else:
            if i != total_rows - 1:
                height_ratios.append(0.15)

    # For columns, add an extra gap after every column
    width_ratios = []
    for j in range(total_cols):
        width_ratios.append(1)
        if j != total_cols - 1:  # after each column except last
            width_ratios.append(0.10)  # horizontal gap spacing

    # Define figure and gridspec with custom spacing
    if is_mainEff_or_core2wayEpistasis:
        fig = plt.figure(figsize=(4*len(n_values) * 1.15, 3*len(h_values)*2 * 1.1))
    else:
        fig = plt.figure(figsize=(4*len(n_values) * 1.15, 3*len(h_values) * 1.1))
    gs = gridspec.GridSpec(
        nrows=len(height_ratios),
        ncols=len(width_ratios),
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.25,  # fine-tune base spacing
        wspace=0.25
    )

    # Build axes array only in the actual plot cells (skip gap cells)
    axes = np.empty((total_rows, total_cols), dtype=object)
    row_ptr, col_ptr = 0, 0
    for i in range(len(height_ratios)):
        if is_mainEff_or_core2wayEpistasis:
            if i % 3 == 2:  # skip gap row
                continue
        else:
            if i % 2 == 1:  # skip gap row
                continue
        col_ptr = 0
        for j in range(len(width_ratios)):
            if j % 2 == 1:  # skip gap column
                continue
            axes[row_ptr, col_ptr] = fig.add_subplot(gs[i, j])
            col_ptr += 1
        row_ptr += 1
    # --- END NEW ATTEMPT

    if len(h_values)*2 == 1 and len(n_values) == 1:
        axes = np.array([[axes]])  # ensure 2D

    xtick_labels = ['Optimal','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
    # for i, h in enumerate(h_values):
    # flipped so that y-axis is shown in ascending order from bottom to top
    for i, h in enumerate(sorted(h_values, reverse=True)):
        for j, n in enumerate(n_values):
            if is_mainEff_or_core2wayEpistasis:
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
                    # sns.heatmap(df1, ax=ax_bot, annot=False, cmap=custom_cmap, cbar=False, xticklabels=xtick_labels, yticklabels=False)
                    sns.heatmap(df1, ax=ax_bot, annot=False, cmap=custom_cmap, cbar=False, xticklabels=False, yticklabels=False)
                else:
                    ax_bot.axis('off')

                # if j==0:
                #     ax_top.set_ylabel("E", rotation=0, labelpad=20, fontsize=12)
                #     ax_bot.set_ylabel("H", rotation=0, labelpad=20, fontsize=12)
                # Put "E" and "H" labels on the right side of the heatmaps in the last column
                if j == len(n_values) - 1:
                    # maybe add if clauses in the event that a dataset does not have EDM-1 or EDM-2
                    ax_top.set_ylabel("E", rotation=0, labelpad=20, fontsize=18)
                    ax_top.yaxis.set_label_position("right")
                    
                    ax_bot.set_ylabel("H", rotation=0, labelpad=20, fontsize=18)
                    ax_bot.yaxis.set_label_position("right")
            else:
                ax = axes[i, j]

                # Find the value in the dict corresponding to this n and h (could be either EDM-1 or EDM-2 depending on dataset, but only one of them)
                df = next(
                    data_dict[key] 
                    for key in data_dict 
                    if key[0] == n and key[1] == h
                )

                if df is not None:
                    sns.heatmap(df, ax=ax, annot=False, cmap=custom_cmap, cbar=False, xticklabels=False, yticklabels=False)
                else:
                    ax.axis('off')
                
                if j == len(n_values) - 1:
                    edm_value = next(key[2] for key in data_dict if key[0] == n and key[1] == h)
                    if edm_value == '1':
                        ax.set_ylabel("H", rotation=0, labelpad=20, fontsize=18)
                    elif edm_value == '2':
                        ax.set_ylabel("E", rotation=0, labelpad=20, fontsize=18)

                    ax.yaxis.set_label_position("right")
                    


    # # Add one colorbar on the far right
    # cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    # # take any df to draw dummy heatmap for cbar
    # any_df = next(iter(data_dict.values()))
    # sns.heatmap(any_df, cmap=custom_cmap, cbar_ax=cbar_ax, cbar_kws={'label': 'Power (Frequency of Success)'}, annot=False)

    # # fig.suptitle("Unified Heatmaps", fontsize=16)
    # fig.supxlabel("Number of Training Instances (n)", fontsize=14)
    # fig.supylabel("Heritability of Model", fontsize=14)

    # plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    # Set x-axis labels for Number of Training Instances
    for j, n in enumerate(n_values):
        # Place label centered below the corresponding column (under the last row for that column)
        mid_axs = axes[-1, j] if len(h_values)*2 > 1 else axes[1, j]
        mid_axs.set_xlabel(str(n), fontsize=20)
        mid_axs.xaxis.set_label_coords(0.5, -0.2)  # adjust vertical padding

    # Set y-axis labels for Heritability of Model (once per heritability row)
    # for i, h in enumerate(h_values):
    for i, h in enumerate(sorted(h_values, reverse=True)):
        if is_mainEff_or_core2wayEpistasis:
            # First column only
            ax_top = axes[i*2, 0]
            ax_bot = axes[i*2 + 1, 0]
            
            # Position the label in the middle of top and bottom heatmaps
            mid_y = 0  # normalized vertical coordinate (0 = bottom of ax, 1 = top of ax)
            
            # if there is only one column, can't use set_ylabel twice on the same axis (will overwrite the first one, "E"); so use .text instead
            if len(n_values) == 1:
                ax_top.text(-0.2, mid_y - 0.1, str(h), rotation=0, fontsize=20, va='center', ha='center', transform=ax_top.transAxes)
            else:
                # Use the top subplot to place the label vertically centered
                ax_top.set_ylabel(str(h), rotation=0, fontsize=20)
                ax_top.yaxis.set_label_coords(-0.2, mid_y)
        else:
            # First column only
            ax = axes[i, 0]

            mid_y = 0.5

            if len(n_values) == 1:
                ax.text(-0.2, mid_y, str(h), rotation=0, fontsize=20, va='center', ha='center', transform=ax.transAxes)
            else:
                ax.set_ylabel(str(h), rotation=0, fontsize=20)
                ax.yaxis.set_label_coords(-0.2, mid_y)

    # # Total rows and columns
    # total_rows = len(h_values)*2
    # total_cols = len(n_values)

    # # Draw vertical lines between columns (n-values)
    # for j in range(1, total_cols):
    #     # Get right edge of previous column and left edge of next column
    #     left_bbox = axes[0, j-1].get_position()
    #     right_bbox = axes[0, j].get_position()
    #     x = (left_bbox.x1 + right_bbox.x0) / 2  # middle between subplots
    #     line = mlines.Line2D([x, x], [0, 1], transform=fig.transFigure, color='black', linewidth=1.5)
    #     fig.add_artist(line)

    # # Draw horizontal lines between heritabilities (h-values)
    # for i in range(1, len(h_values)):
    #     # Bottom of the bottom subplot in previous heritability
    #     prev_bottom_bbox = axes[(i-1)*2 + 1, 0].get_position()
    #     # Top of the top subplot in current heritability
    #     curr_top_bbox = axes[i*2, 0].get_position()
        
    #     # Midpoint between these two edges
    #     y = (prev_bottom_bbox.y0 + curr_top_bbox.y1) / 2
        
    #     line = mlines.Line2D([0, 1], [y, y], transform=fig.transFigure, color='black', linewidth=1.5)
    #     fig.add_artist(line)

    # Adjust global labels with extra padding
    fig.supxlabel("Number of Training Instances (n)", fontsize=22, x=0.5, y=0.02)
    # fig.canvas.draw()
    # renderer = fig.canvas.get_renderer()
    # bbox_axes = axes[0,0].get_position()
    # label_x = bbox_axes.x0 - 0.05  # offset by ~5% of figure width
    # fig.supylabel("Heritability of Model", fontsize=22, x=label_x, y=0.5)
    if is_xor:
        fig.supylabel("Number of Predictive Features", fontsize=22, x=0.02, y=0.5)
    elif is_mainEffadditive_or_2wayEpiHet:
        fig.supylabel("Proportion of Instances from Subgroup 1", fontsize=22, x=0.02, y=0.5)
    else:
        fig.supylabel("Heritability of Model", fontsize=22, x=0.02, y=0.5)

    # Tight layout with extra spacing
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    # plt.tight_layout(rect=[0.05, 0.05, 0.85, 0.95])

    # # CODE TO DRAW TICK MARK SEPARATORS:
    # # Number of rows and columns in the grid
    # total_rows = len(h_values) * 2
    # total_cols = len(n_values)

    # # --- X-axis tick marks (vertical) ---
    # for j in range(total_cols - 1):  # exclude last column
    #     # Right edge of column j
    #     bbox_right = axes[0, j].get_position().x1  # top row
    #     bbox_right_bottom = axes[-1, j].get_position().x1  # bottom row
        
    #     # Top tick (first row)
    #     fig.add_artist(mlines.Line2D([bbox_right, bbox_right], 
    #                                 [axes[0, j].get_position().y1, axes[0, j].get_position().y1 + 0.02],
    #                                 transform=fig.transFigure, color='black', linewidth=4))
    #     # Bottom tick (last row)
    #     fig.add_artist(mlines.Line2D([bbox_right_bottom, bbox_right_bottom], 
    #                                 [axes[-1, j].get_position().y0 - 0.02, axes[-1, j].get_position().y0],
    #                                 transform=fig.transFigure, color='black', linewidth=4))

    # # --- Y-axis tick marks (horizontal) ---
    # for i in range(1, total_rows - 2, 2):  # every 2 rows
    #     # Bottom edge of row i (top of next heritability)
    #     bbox_bottom_left = axes[i, 0].get_position().y0
    #     bbox_bottom_right = axes[i, -1].get_position().y0
        
    #     # Left tick (first column)
    #     fig.add_artist(mlines.Line2D([axes[i, 0].get_position().x0 - 0.02, axes[i, 0].get_position().x0], 
    #                                 [bbox_bottom_left, bbox_bottom_left], 
    #                                 transform=fig.transFigure, color='black', linewidth=4))
    #     # Right tick (last column)
    #     fig.add_artist(mlines.Line2D([axes[i, -1].get_position().x1, axes[i, -1].get_position().x1 + 0.02], 
    #                                 [bbox_bottom_right, bbox_bottom_right], 
    #                                 transform=fig.transFigure, color='black', linewidth=4))

    # ** LINES BETWEEN INSTANCE/HERITABILITY COMBINATIONS:
    # --- DRAW DIVIDER LINES THROUGH GAP COLUMNS AND ROWS ---
    # Use figure coordinates (0–1 range)
    for j in range(total_cols - 1):
        # Compute midpoint between the right edge of column j and left edge of next column
        left_bbox = axes[0, j].get_position()
        right_bbox = axes[0, j+1].get_position()
        x_mid = (left_bbox.x1 + right_bbox.x0) / 2
        y_bottom = axes[-1,0].get_position().y0
        y_top = axes[0,0].get_position().y1

        # Vertical line (spanning entire figure)
        line = mlines.Line2D([x_mid, x_mid], [y_bottom, y_top],
                            transform=fig.transFigure, color='black', linewidth=1.5)
        fig.add_artist(line)

    if is_mainEff_or_core2wayEpistasis:
        # Horizontal dividers after every heritability block (every 2 rows)
        for i in range(1, len(h_values)):
            # Get bounding boxes for last heatmap in block i-1 and first in block i
            prev_bottom = axes[(i-1)*2 + 1, 0].get_position().y0  # bottom of bottom subplot of previous block
            next_top = axes[i*2, 0].get_position().y1             # top of top subplot of next block
            y_mid = (prev_bottom + next_top) / 2
            x_left = axes[0,0].get_position().x0
            x_right = axes[0,-1].get_position().x1

            # Horizontal line (spanning entire figure)
            line = mlines.Line2D([x_left, x_right], [y_mid, y_mid],
                                transform=fig.transFigure, color='black', linewidth=1.5)
            fig.add_artist(line)
    else:
        for i in range(0, len(h_values) - 1):
            prev_bottom = axes[i, 0].get_position().y0  # bottom of bottom subplot of previous block
            next_top = axes[i+1, 0].get_position().y1             # top of top subplot of next block
            y_mid = (prev_bottom + next_top) / 2
            x_left = axes[0,0].get_position().x0
            x_right = axes[0,-1].get_position().x1

            # Horizontal line (spanning entire figure)
            line = mlines.Line2D([x_left, x_right], [y_mid, y_mid],
                                transform=fig.transFigure, color='black', linewidth=1.5)
            fig.add_artist(line)
        

    outdir = os.path.basename(os.path.normpath(args.basedir))
    parentdir = os.path.dirname(os.path.normpath(args.basedir))
    # save_path = os.path.join(parentdir, outdir, args.prefix + "unified_heatmaps.pdf")
    # save_path = os.path.join(parentdir, outdir, args.prefix + "unified_heatmaps_MultiSWRFDBtest.pdf")
    # save_path = os.path.join(parentdir, outdir, args.prefix + "unified_heatmaps_MultiSWRFDBtest_with3sd.pdf")
    save_path = os.path.join(parentdir, outdir, args.prefix + "unified_heatmaps_withMuRelief.pdf")
    # save_path = os.path.join(parentdir, outdir, args.prefix + "unified_heatmaps_validcheck.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Unified heatmap saved to {save_path}")

if __name__ == "__main__":
    main()