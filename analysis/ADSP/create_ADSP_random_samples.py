import pandas as pd

def main():
    train_data = pd.read_csv('/common/moorej3lab_ADSP_WGS/matched_3splits_ids/train_final.csv')
    val_data = pd.read_csv('/common/moorej3lab_ADSP_WGS/matched_3splits_ids/validation_final.csv')

    # concatenate train and val data to create a set of 15k instances
    train_val_combined = pd.concat([train_data, val_data])

    # randomly shuffle the order of the 15k instances
    train_val_combined = train_val_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # create stratified random samples (50% case proportion) with 2000 instances each
    for i in range(1, 11):
        sample = (
            train_val_combined
            .groupby('outcome', group_keys=False)
            .sample(
                n=1000,  # This is per group (i.e. 0 and 1)
                replace=False,
                random_state=i
            )
        )
        sample.to_csv(
            f'/common/kazemik/ReBATE_ADSP/ADSP_traindata_samples/ADSP_randomsample_seed{i}.txt',
            index=False
        )