from core_tbi.data.synthetic import FEATURES, make_synthetic_longitudinal_tbi
from core_tbi.models.baseline import RecoveryPlaneModel


def test_recovery_plane_scores_and_states():
    frame = make_synthetic_longitudinal_tbi(n_animals_per_condition=4, sessions_per_timepoint=3)
    scored = RecoveryPlaneModel(FEATURES).fit_transform(frame)
    assert set(scored.recovery_state) <= {"restitution", "compensation", "persistent_dysfunction", "uncertain"}
    assert scored.counterfactual_deviation.notna().all()
    assert (scored.loc[(scored.condition == "tbi") & (scored.timepoint == "day_10"), "recovery_state"] == "compensation").any()
