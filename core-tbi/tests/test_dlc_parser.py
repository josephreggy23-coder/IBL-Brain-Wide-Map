import pandas as pd
from core_tbi.data.dlc_parser import parse_dlc_csv


def test_flattened_dlc_parser(tmp_path):
    source = tmp_path / "pose.csv"
    pd.DataFrame({"nose_x": [1, 2], "nose_y": [3, 4], "nose_likelihood": [.9, .8]}).to_csv(source, index=False)
    result = parse_dlc_csv(source, {"animal_id": "A", "dataset_id": "demo", "session_id": "s"}, 10)
    assert list(result.bodypart.unique()) == ["nose"]
    assert result.time_seconds.tolist() == [0.0, 0.1]
