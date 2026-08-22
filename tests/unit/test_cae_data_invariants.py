from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.pipelines.multi_stage_ae.cae_pipeline import run_keras_cae_pipeline


def test_partition_label_isolation(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mock_mvtec"
    category_dir = dataset_dir / "bottle"

    train_good_dir = category_dir / "train" / "good"
    test_good_dir = category_dir / "test" / "good"
    test_defect_dir = category_dir / "test" / "defect"
    ground_truth_dir = category_dir / "ground_truth" / "defect"

    for d in [train_good_dir, test_good_dir, test_defect_dir, ground_truth_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def create_image(path: Path, size: tuple[int, int] = (16, 16)) -> None:
        img = Image.new("RGB", size, color=(128, 128, 128))
        img.save(path)

    def create_mask(path: Path, size: tuple[int, int] = (16, 16)) -> None:
        img = Image.new("L", size, color=255)
        img.save(path)

    for i in range(20):
        create_image(train_good_dir / f"{i:03d}.png")
    for i in range(10):
        create_image(test_good_dir / f"{i:03d}.png")
    for i in range(10):
        create_image(test_defect_dir / f"{i:03d}.png")
        create_mask(ground_truth_dir / f"{i:03d}_mask.png")

    import sklearn.model_selection as ms

    original_split = ms.train_test_split
    split_records = []

    def mock_train_test_split(*arrays, **options):
        result = original_split(*arrays, **options)
        split_records.append({"arrays": arrays, "result": result, "options": options})
        return result

    class AbortPipeline(Exception):
        pass

    with patch("sklearn.model_selection.train_test_split", side_effect=mock_train_test_split):
        with patch("app.pipelines.multi_stage_ae.cae_pipeline.build_cae", side_effect=AbortPipeline):
            try:
                run_keras_cae_pipeline(
                    data_root=str(dataset_dir),
                    category="bottle",
                    img_size=32,
                    crop_size=16,
                    crop_stride=16,
                    latent_channels=8,
                    epochs=1,
                    batch_size=2,
                    force_retrain=True,
                )
            except AbortPipeline:
                pass

    assert len(split_records) == 1
    rec = split_records[0]
    assert rec["options"].get("test_size") == 0.15
    assert len(rec["arrays"][0]) == 20
    assert len(rec["result"][0]) == 17
    assert len(rec["result"][1]) == 3

    train_paths = set(rec["result"][0]["path"].tolist())
    val_paths = set(rec["result"][1]["path"].tolist())
    assert len(train_paths.intersection(val_paths)) == 0


def test_augmentation_leakage_prevention(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mock_mvtec"
    category_dir = dataset_dir / "bottle"

    for d in ["train/good", "test/good", "test/defect", "ground_truth/defect"]:
        (category_dir / d).mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    for i in range(1, 10):
        img.save(category_dir / "train" / "good" / f"{i:03d}.png")
    img.save(category_dir / "test" / "good" / "000.png")
    img.save(category_dir / "test" / "defect" / "000.png")
    Image.new("L", (32, 32), color=255).save(category_dir / "ground_truth" / "defect" / "000_mask.png")

    import app.pipelines.multi_stage_ae.cae_pipeline as p_mod

    original_augment = p_mod.augment_batch
    augment_calls = []

    def mock_augment_batch(batch, augmenter):
        augment_calls.append(len(batch))
        return original_augment(batch, augmenter)

    class AbortPipeline(Exception):
        pass

    with patch("app.pipelines.multi_stage_ae.cae_pipeline.augment_batch", side_effect=mock_augment_batch):
        with patch("app.pipelines.multi_stage_ae.cae_pipeline.build_cae", side_effect=AbortPipeline):
            try:
                run_keras_cae_pipeline(
                    data_root=str(dataset_dir),
                    category="bottle",
                    img_size=32,
                    crop_size=16,
                    crop_stride=16,
                    latent_channels=8,
                    epochs=1,
                    batch_size=2,
                    force_retrain=True,
                )
            except AbortPipeline:
                pass

    assert sum(augment_calls) == 7
