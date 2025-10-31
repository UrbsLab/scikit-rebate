# job_global_rba_rankings.py
import os
import argparse
import pandas as pd

def collect_rankings(root_dir, include_subdirs=None):
    """
    Collects all rankings_list.csv files from subdirectories under root_dir.
    Optionally filter which subdirectories to include.
    """
    all_dfs = []
    
    for sub in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        # If --include is specified, skip subdirs not in the list
        if include_subdirs and sub not in include_subdirs:
            continue

        rankings_path = os.path.join(sub_path, 'rankings_list.csv')
        if not os.path.exists(rankings_path):
            print(f"[WARN] rankings_list.csv not found in {sub_path}, skipping.")
            continue

        try:
            df = pd.read_csv(rankings_path, comment='#')  # ignore comment line with title
        except Exception as e:
            print(f"[ERROR] Could not read {rankings_path}: {e}")
            continue

        all_dfs.append(df)
        print(f"[INFO] Loaded rankings_list.csv from {sub}")

    if not all_dfs:
        print("[ERROR] No valid rankings_list.csv files found.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def compute_global_summary(all_rankings_df):
    """
    Computes global mean and median rank per RBA.
    """
    summary = (
        all_rankings_df.groupby('RBA')['Rank']
        .agg(['mean', 'median'])
        .reset_index()
        .rename(columns={'mean': 'Mean', 'median': 'Median'})
    )
    summary.sort_values(by=['Mean', 'Median'], inplace=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate global rankings from multiple subdirectories.")
    parser.add_argument("root_dir", help="Top-level directory containing dataset subdirectories.")
    parser.add_argument("--include", nargs="+", default=None,
                        help="Optional list of subdirectories to include (default: all).")

    args = parser.parse_args()
    root_dir = args.root_dir

    # short names -> actual subdirectory names
    SHORT_TO_FULL_SUBDIR = {
        "maineff": "GAMETES_2.2_dev_peter_mainEff_Datasets_Loc_1_Qnt_2_Pop_100000",
        "core2wayepistasis": "GAMETES_2.2_dev_peter_core2wayEpistasis_Datasets_Loc_2_Qnt_2_Pop_100000",
        "xor": "GAMETES_2.2_dev_peter_XOR",
        "maineff_2": "GAMETES_2.2_dev_peter_mainEff_additive_2_Datasets_2Het_Loc_1_Qnt_2_Pop_100000",
        "maineff_4": "GAMETES_2.2_dev_peter_mainEff_additive_4_Datasets_2Het_Loc_1_Qnt_2_Pop_100000",
        "2wayepi": "GAMETES_2.2_dev_peter_2wayEpiHeterogeneity_Datasets_2Het_Loc_2_Qnt_2_Pop_100000",
        "epi_order": "GAMETES_2.2_dev_peter_epi_order_Datasets_Loc_3_Qnt_2_Pop_100000"
    }

    user_include = args.include

    if user_include:
        # map the user-specified short names to actual subdirectory names
        include_subdirs = []
        for short_name in user_include:
            # full_name = SHORT_TO_FULL_SUBDIR.get(short_name)
            full_name = SHORT_TO_FULL_SUBDIR.get(short_name.lower())
            if full_name:
                include_subdirs.append(full_name)
            else:
                print(f"[WARN] No mapping found for '{short_name}', skipping.")
    else:
        include_subdirs = None  # default: include all subdirectories
    
    print(f"[INFO] Subdirectories to include: {include_subdirs}")

    combined_df = collect_rankings(root_dir, include_subdirs)
    if combined_df is None:
        return

    # --- Save global concatenated rankings_list.csv ---
    global_rankings_path = os.path.join(root_dir, 'global_rankings_list.csv')
    combined_df.to_csv(global_rankings_path, index=False)
    print(f"[INFO] Saved global rankings list: {global_rankings_path}")

    # --- Compute and save global RBA rankings ---
    global_summary_df = compute_global_summary(combined_df)
    global_summary_path = os.path.join(root_dir, 'global_rba_rankings.csv')
    global_summary_df.to_csv(global_summary_path, index=False)
    print(f"[INFO] Saved global RBA rankings: {global_summary_path}")


if __name__ == "__main__":
    main()