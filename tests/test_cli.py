from decision_geometry.cli import build_parser


def test_cli_accepts_a_reproducibility_seed():
    arguments = build_parser().parse_args(["--seed", "42"])
    assert arguments.seed == 42
