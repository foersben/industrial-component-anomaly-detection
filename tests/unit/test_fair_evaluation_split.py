"""Tests for the shared fair baseline-evaluation data protocol."""

from pathlib import Path

import pandas as pd
import pytest

from app.domain.data import FAIR_EVALUATION_PROTOCOL, build_fair_evaluation_split


def _manifest(tmp_path: Path, train_count: int = 209, test_count: int = 83) -> pd.DataFrame:
    rows = [
        {
            "path": str(tmp_path / "bottle" / "train" / "good" / f"{index:03}.png"),
            "product": "bottle",
            "split": "train",
            "is_anomaly": False,
        }
        for index in range(train_count)
    ]
    rows.extend(
        {
            "path": str(tmp_path / "bottle" / "test" / ("good" if index < 20 else "broken") / f"{index:03}.png"),
            "product": "bottle",
            "split": "test",
            "is_anomaly": index >= 20,
        }
        for index in range(test_count)
    )
    return pd.DataFrame(rows)


def test_shared_split_is_deterministic_disjoint_and_exhaustive(tmp_path: Path) -> None:
    """The shared split is stable, disjoint, exhaustive, and has bottle counts."""
    manifest = _manifest(tmp_path)
    shuffled_train = manifest.loc[manifest["split"] == "train"].sample(frac=1.0, random_state=7)
    stable_test = manifest.loc[manifest["split"] == "test"]
    shuffled_manifest = pd.concat([shuffled_train, stable_test], ignore_index=True)

    first = build_fair_evaluation_split(manifest, "bottle")
    second = build_fair_evaluation_split(shuffled_manifest, "bottle")

    assert first.fitting_paths == second.fitting_paths
    assert first.validation_paths == second.validation_paths
    assert first.test_paths == second.test_paths
    assert first.test_paths == stable_test["path"].tolist()
    assert first.evidence() == second.evidence()
    assert len(first.fitting_paths) == 177
    assert len(first.validation_paths) == 32
    assert len(first.test_paths) == 83
    assert not set(first.fitting_paths) & set(first.validation_paths)
    assert set(first.fitting_paths) | set(first.validation_paths) == set(
        manifest.loc[manifest["split"] == "train", "path"]
    )
    assert first.evidence()["protocol"] == FAIR_EVALUATION_PROTOCOL


def test_shared_split_digest_changes_when_membership_changes(tmp_path: Path) -> None:
    """Ordered path digests identify changes to a protocol partition."""
    manifest = _manifest(tmp_path, train_count=20, test_count=5)
    original = build_fair_evaluation_split(manifest, "bottle")
    manifest.loc[manifest["split"] == "test", "path"] = [
        f"{path}.changed" for path in manifest.loc[manifest["split"] == "test", "path"]
    ]

    changed = build_fair_evaluation_split(manifest, "bottle")

    assert original.fitting_digest == changed.fitting_digest
    assert original.validation_digest == changed.validation_digest
    assert original.test_digest != changed.test_digest


def test_shared_split_rejects_invalid_manifest(tmp_path: Path) -> None:
    """A malformed manifest fails visibly instead of creating partial evidence."""
    with pytest.raises(ValueError, match="required columns"):
        build_fair_evaluation_split(pd.DataFrame({"path": [str(tmp_path / "x.png")]}), "bottle")
