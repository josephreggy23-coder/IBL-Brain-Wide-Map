from core_tbi.registry import load_registry


def test_registry_has_primary_dataset():
    assert "alma" in load_registry()
