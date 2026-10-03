import json
from pathlib import Path

MANIFEST = json.loads(
    (Path(__file__).resolve().parents[1] / "configs" / "gate_subset.json").read_text()
)


def test_gate_subset_is_fixed_sized_and_disjoint_from_harvest() -> None:
    assert len(MANIFEST) == 6
    for name, split in MANIFEST.items():
        assert len(split["gate"]) == 100, name
        assert len(split["harvest"]) == 50, name
        assert not set(split["gate"]) & set(split["harvest"]), name
