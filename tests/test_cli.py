import pytest

from decision_geometry.cli import build_parser


def test_cli_accepts_a_reproducibility_seed():
    arguments = build_parser().parse_args(["--seed", "42"])
    assert arguments.seed == 42


@pytest.mark.parametrize("option", ["--bin-size", "--max-units"])
def test_cli_rejects_non_positive_analysis_settings(option):
    with pytest.raises(SystemExit):
        build_parser().parse_args([option, "0"])
