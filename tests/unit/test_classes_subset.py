"""Training on a subset of a dataset's classes, without editing annotation
files (see repo-local research_class.md for the design writeup).

Generalizes the existing single_cls remap-at-parse-time mechanism: classes=
picks which original dataset class ids survive; every other class's boxes
are dropped as if never annotated, source annotation files stay untouched.
Kept ids are NOT compacted to a contiguous range -- they keep their original
numbering, so predictions and checkpoint metadata stay directly comparable
to the full dataset (an external COCO-style evaluator, an exported model
read by raw class index, a person who knows "id 7 = train") without any
translation layer. The cost is that the model head still covers every index
up to the highest kept id; unrequested classes below it simply never
receive positive supervision, the same as a class with zero occurrences in
an ordinary dataset. single_cls stays a genuine remap (collapses every kept
id to 0) since that is an intentional merge, not an exclusion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from libreyolo.data.dataset import COCODataset, YOLODataset
from libreyolo.data.obb import parse_yolo_obb_label_line
from libreyolo.data.utils import build_class_remap, load_data_config
from libreyolo.data.yolo_coco_api import parse_yolo_label_line
from libreyolo.training.config import TrainConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# TrainConfig validation
# ---------------------------------------------------------------------------


def test_train_config_classes_defaults_off():
    assert TrainConfig().classes is None


def test_train_config_classes_normalizes_to_int_list():
    assert TrainConfig(classes=[0, "1", 3]).classes == [0, 1, 3]


@pytest.mark.parametrize(
    "bad",
    [[], [-1, 0], [0, 0, 1]],
)
def test_train_config_classes_rejects_invalid(bad):
    with pytest.raises(ValueError):
        TrainConfig(classes=bad)


# ---------------------------------------------------------------------------
# build_class_remap / load_data_config
# ---------------------------------------------------------------------------


def test_build_class_remap_none_when_no_classes():
    assert build_class_remap(None) is None


def test_build_class_remap_keeps_original_ids():
    assert build_class_remap([3, 0, 1]) == {0: 0, 1: 1, 3: 3}


def test_build_class_remap_single_cls_collapses_to_zero():
    assert build_class_remap([3, 0, 1], single_cls=True) == {0: 0, 1: 0, 3: 0}


NAMES10 = (
    "car",
    "bicycle",
    "person",
    "dog",
    "truck",
    "cat",
    "bus",
    "train",
    "boat",
    "bird",
)


def _write_data_yaml(tmp_path: Path, nc=4, names=("car", "bicycle", "dog", "cat")):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/train",
                "nc": nc,
                "names": list(names),
            }
        )
    )
    return data_yaml


def test_load_data_config_classes_leaves_nc_and_names_untouched(tmp_path):
    data_yaml = _write_data_yaml(tmp_path)

    cfg = load_data_config(str(data_yaml), autodownload=False, classes=[0, 1, 3])

    assert cfg["nc"] == 4
    assert cfg["names"] == ["car", "bicycle", "dog", "cat"]
    assert cfg["_class_remap"] == {0: 0, 1: 1, 3: 3}
    assert "_original_nc" not in cfg
    assert "_original_names" not in cfg


def test_load_data_config_classes_and_single_cls_collapse_kept_to_one(tmp_path):
    data_yaml = _write_data_yaml(tmp_path)

    cfg = load_data_config(
        str(data_yaml), autodownload=False, classes=[0, 1, 3], single_cls=True
    )

    assert cfg["nc"] == 1
    assert cfg["names"] == {0: "object"}
    assert cfg["_class_remap"] == {0: 0, 1: 0, 3: 0}
    assert cfg["_original_nc"] == 4
    assert cfg["_original_names"] == ["car", "bicycle", "dog", "cat"]


def test_load_data_config_classes_out_of_range_fails_fast(tmp_path):
    data_yaml = _write_data_yaml(tmp_path)

    with pytest.raises(ValueError, match="outside this dataset's declared nc"):
        load_data_config(str(data_yaml), autodownload=False, classes=[0, 99])


# ---------------------------------------------------------------------------
# Label parsers
# ---------------------------------------------------------------------------


def test_parse_yolo_label_line_excluded_class_is_silently_dropped():
    remap = {0: 0, 1: 1, 3: 3}
    line = "2 0.5 0.5 0.2 0.2"  # dog, excluded

    assert parse_yolo_label_line(line, 100, 100, None, class_remap=remap) is None


def test_parse_yolo_label_line_kept_class_keeps_its_original_id():
    remap = {0: 0, 1: 1, 3: 3}
    line = "3 0.5 0.5 0.2 0.2"  # cat, id 3, kept as 3

    result = parse_yolo_label_line(line, 100, 100, None, class_remap=remap)

    assert result[0] == 3


def test_parse_yolo_obb_label_line_excluded_class_raises():
    remap = {0: 0, 1: 1, 3: 3}
    row = "2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"

    with pytest.raises(ValueError, match="excluded by classes"):
        parse_yolo_obb_label_line(row, class_remap=remap)


def test_parse_yolo_obb_label_line_kept_class_keeps_its_original_id():
    remap = {0: 0, 1: 1, 3: 3}
    row = "3 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"

    cls_id, _ = parse_yolo_obb_label_line(row, class_remap=remap)

    assert cls_id == 3


# ---------------------------------------------------------------------------
# YOLODataset (.txt)
# ---------------------------------------------------------------------------


def _write_yolo_sample(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    label_path = tmp_path / "sample.txt"
    Image.new("RGB", (64, 48), color="white").save(image_path)
    label_path.write_text(
        "0 0.1 0.1 0.1 0.1\n1 0.3 0.3 0.1 0.1\n2 0.5 0.5 0.1 0.1\n3 0.7 0.7 0.1 0.1\n",
        encoding="utf-8",
    )
    return [image_path], [label_path]


def test_yolo_dataset_class_remap_drops_excluded_keeps_original_ids(tmp_path):
    image_files, label_files = _write_yolo_sample(tmp_path)
    remap = {0: 0, 1: 1, 3: 3}  # drop original id 2 (dog); keep the rest as-is

    dataset = YOLODataset(
        img_files=image_files,
        label_files=label_files,
        img_size=(64, 64),
        class_remap=remap,
    )

    labels = dataset.annotations[0][0]
    assert labels.shape[0] == 3
    assert sorted(labels[:, 4].tolist()) == [0.0, 1.0, 3.0]


# ---------------------------------------------------------------------------
# COCODataset (JSON)
# ---------------------------------------------------------------------------


def _write_coco_sample(tmp_path: Path):
    image_dir = tmp_path / "images" / "train"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    Image.new("RGB", (64, 48), color="white").save(image_dir / "sample.jpg")
    (annotation_dir / "train.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "sample.jpg", "width": 64, "height": 48}
                ],
                "annotations": [
                    {
                        "id": i,
                        "image_id": 1,
                        "category_id": cat_id,
                        "bbox": [4, 4, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    }
                    for i, cat_id in enumerate((1, 2, 3, 4), start=1)
                ],
                "categories": [
                    {"id": 1, "name": "car"},
                    {"id": 2, "name": "bicycle"},
                    {"id": 3, "name": "dog"},
                    {"id": 4, "name": "cat"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_coco_dataset_classes_drops_excluded_keeps_original_labels(tmp_path):
    pytest.importorskip("pycocotools")
    _write_coco_sample(tmp_path)

    dataset = COCODataset(
        data_dir=str(tmp_path),
        json_file="annotations/train.json",
        name="images/train",
        img_size=(64, 64),
        num_classes=4,
        names={0: "car", 1: "bicycle", 2: "dog", 3: "cat"},
        classes=[0, 1, 3],
    )

    # category 3 (dog, label 2) is excluded; the rest keep their original label.
    assert dataset.category_id_to_label == {1: 0, 2: 1, 4: 3}
    # names cover the full original set, including the excluded class --
    # nothing is hidden, it just never receives positive supervision.
    assert dataset._classes == ("car", "bicycle", "dog", "cat")
    labels = dataset.annotations[0][0]
    assert labels.shape[0] == 3
    assert sorted(labels[:, 4].tolist()) == [0.0, 1.0, 3.0]
