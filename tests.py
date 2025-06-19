import pandas as pd
import numpy as np
import pytest
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, TURF, SWRFstar, SWRFstar2, SWRF, TBD1star, TBD1, TBD2star, TBD2
import warnings

warnings.filterwarnings('ignore')
np.random.seed(3249083)

# Load datasets once for all tests
datasets = {
    "binary": pd.read_csv('data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv').sample(frac=1),
    "cont_endpoint": pd.read_csv('data/GAMETES_Epistasis_2-Way_continuous_endpoint_a_20s_1600her_0.4__maf_0.2_EDM-2_01.csv').sample(frac=1),
    "mixed": pd.read_csv('data/GAMETES_Epistasis_2-Way_mixed_attribute_a_20s_1600her_0.4__maf_0.2_EDM-2_01.csv').sample(frac=1),
    "missing": pd.read_csv('data/GAMETES_Epistasis_2-Way_missing_values_0.1_a_20s_1600her_0.4__maf_0.2_EDM-2_01.csv').sample(frac=1),
    "multiclass": pd.read_csv('data/3Class_Datasets_Loc_2_01.csv').sample(frac=1)
}

# Parametrize over selectors
selectors = [
    # Existing selectors (commented out for brevity)
    # (ReliefF, {'n_features_to_select': 5, 'n_neighbors': 1}), 
    # (ReliefF, {'n_features_to_select': 5, 'n_neighbors': 10}),
    # (SURF, {'n_features_to_select': 5}),
    # (SURFstar, {'n_features_to_select': 5}),
    # (MultiSURF, {'n_features_to_select': 5}),
    # (MultiSURFstar, {'n_features_to_select': 5}),
    # (TURF, {'relief_object': ReliefF(n_features_to_select=5, n_neighbors=10), 'pct': 0.5, 'num_scores_to_return': 5}),
    # New algorithms
    (SWRFstar, {'n_features_to_select': 5}),
    (SWRFstar2, {'n_features_to_select': 5}),
    (SWRF, {'n_features_to_select': 5}),
    (TBD1star, {'n_features_to_select': 5}),
    (TBD1, {'n_features_to_select': 5}),
    (TBD2star, {'n_features_to_select': 5}),
    (TBD2, {'n_features_to_select': 5}),
]

@pytest.mark.parametrize("dataset_name,genetic_data", datasets.items())
@pytest.mark.parametrize("cls,kwargs", selectors)
def test_selector_on_dataset(cls, kwargs, dataset_name, genetic_data):
    features = genetic_data.drop('class', axis=1).values
    labels = genetic_data['class'].values
    headers = list(genetic_data.drop("class", axis=1))
    selector = cls(**kwargs)
    selector.fit(features, labels)
    selected_features = [headers[i] for i in selector.top_features_[:5]]
    selected_features_scores = [selector.feature_importances_[i] for i in selector.top_features_[:5]]
    print(f"Selected features: {selected_features}")
    print(f"Feature Scores: {selected_features_scores}")
    assert 'M0P0' in selected_features and 'M0P1' in selected_features, (
        f"{cls.__name__} failed to select M0P0 and M0P1 on {dataset_name} dataset"
    )
    print(f"[PASS] {cls.__name__} on {dataset_name}: Selected M0P0 and M0P1")
