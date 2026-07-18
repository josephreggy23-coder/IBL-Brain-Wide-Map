from core_tbi.data.synthetic import make_synthetic_longitudinal_tbi
from core_tbi.evaluation.splitting import grouped_splits
from core_tbi.evaluation.leakage_checks import assert_no_group_leakage


def test_grouped_splits_keep_animal_together():
    frame = make_synthetic_longitudinal_tbi(n_animals_per_condition=3)
    for train, test in grouped_splits(frame, n_splits=3):
        assert_no_group_leakage(frame, train, test)
