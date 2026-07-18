import pandas as pd
import pytest
from core_tbi.evaluation.leakage_checks import assert_no_group_leakage


def test_leakage_is_rejected():
    frame = pd.DataFrame({"animal_id": ["A", "A", "B"]})
    with pytest.raises(ValueError, match="leakage"):
        assert_no_group_leakage(frame, [0, 2], [1])
