import pandas as pd
from core_tbi.data.alma import split_feature_blocks


def test_split_feature_blocks_keeps_grouped_source_rows():
    raw = pd.DataFrame([[None] * 7 for _ in range(6)])
    raw.iloc[3] = ["group", "stride length (cm)", None, "group", "step height (cm)", None, None]
    raw.iloc[4] = ["baseline", 4.2, None, "day_1", 0.8, None, None]
    raw.iloc[5] = ["day_1", 3.7, None, "day_10", 1.1, None, None]
    result = split_feature_blocks(raw)
    assert len(result) == 4
    assert {"group", "source_block"} <= set(result)
