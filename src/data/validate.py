from src.data.coco import get_default_data_dir, load_cardd_all_splits
from pathlib import Path
from PIL import Image
import json


SPLITS = ("train", "val", "test")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_images_dir(data_dir: Path, split: str) -> Path:
    return data_dir / f"{split}2017"


def split_annotations_path(data_dir: Path, split: str) -> Path:
    return data_dir / "annotations" / f"instances_{split}2017.json"


def validate_structure(data_dir: Path | None = None, splits: tuple[str, ...] = SPLITS) -> list[str]:
    data_dir = data_dir or get_default_data_dir()
    errors = []

    if not data_dir.exists():
        errors.append(f"Missing data directory: {data_dir}")

    annotations_dir = data_dir / "annotations"
    if not annotations_dir.exists():
        errors.append(f"Missing annotations directory: {annotations_dir}")

    for split in splits:
        images_dir = split_images_dir(data_dir, split)
        annotations_path = split_annotations_path(data_dir, split)

        if not images_dir.exists():
            errors.append(f"Missing images directory for split={split}: {images_dir}")

        if not annotations_path.exists():
            errors.append(f"Missing annotations file for split={split}: {annotations_path}")

    return errors


def validate_images(data_dir: Path | None = None, splits: tuple[str, ...] = SPLITS) -> list[str]:
    data_dir = data_dir or get_default_data_dir()
    errors = []

    for split in splits:
        instances = read_json(split_annotations_path(data_dir, split))
        images_dir = split_images_dir(data_dir, split)

        for image_info in instances.get("images", []):
            image_path = images_dir / image_info["file_name"]

            if not image_path.exists():
                errors.append(f"[{split}] Missing image: {image_path}")
                continue

            try:
                with Image.open(image_path) as img:
                    actual_width, actual_height = img.size

                expected_width = image_info["width"]
                expected_height = image_info["height"]

                if actual_width <= 0 or actual_height <= 0:
                    errors.append(f"[{split}] Invalid image size: {image_path}")

                if (actual_width, actual_height) != (expected_width, expected_height):
                    errors.append(
                        f"[{split}] Size mismatch for {image_path}: "
                        f"json=({expected_width}, {expected_height}), "
                        f"actual=({actual_width}, {actual_height})"
                    )

            except Exception as e:
                errors.append(f"[{split}] Cannot open image {image_path}: {e}")

    return errors


def validate_annotations(data_dir: Path | None = None, splits: tuple[str, ...] = SPLITS) -> list[str]:
    data_dir = data_dir or get_default_data_dir()
    errors = []

    for split in splits:
        instances = read_json(split_annotations_path(data_dir, split))

        image_ids = {image["id"] for image in instances.get("images", [])}
        category_ids = {cat["id"] for cat in instances.get("categories", [])}

        for ann in instances.get("annotations", []):
            ann_id = ann.get("id", "<missing_id>")

            if ann.get("image_id") not in image_ids:
                errors.append(f"[{split}] Annotation {ann_id}: unknown image_id={ann.get('image_id')}")

            if ann.get("category_id") not in category_ids:
                errors.append(f"[{split}] Annotation {ann_id}: unknown category_id={ann.get('category_id')}")

            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"[{split}] Annotation {ann_id}: invalid bbox={bbox}")
                continue

            x, y, w, h = bbox

            if w <= 0 or h <= 0:
                errors.append(f"[{split}] Annotation {ann_id}: bbox has non-positive size={bbox}")

            if ann.get("area", 1) <= 0:
                errors.append(f"[{split}] Annotation {ann_id}: non-positive area={ann.get('area')}")

    return errors


def validate_categories(data_dir: Path | None = None, splits: tuple[str, ...] = SPLITS) -> list[str]:
    data_dir = data_dir or get_default_data_dir()
    errors = []

    for split in splits:
        instances = read_json(split_annotations_path(data_dir, split))
        categories = instances.get("categories", [])

        ids = [cat.get("id") for cat in categories]
        names = [cat.get("name") for cat in categories]

        if len(ids) != len(set(ids)):
            errors.append(f"[{split}] Duplicate category ids")

        if len(names) != len(set(names)):
            errors.append(f"[{split}] Duplicate category names")

        for cat in categories:
            if cat.get("id") is None:
                errors.append(f"[{split}] Category without id: {cat}")

            if not cat.get("name"):
                errors.append(f"[{split}] Category without name: {cat}")

    return errors


def validate_bbox_bounds(data_dir: Path | None = None, splits: tuple[str, ...] = SPLITS) -> list[str]:
    data_dir = data_dir or get_default_data_dir()
    errors = []

    for split in splits:
        instances = read_json(split_annotations_path(data_dir, split))

        image_id_to_size = {
            image["id"]: (image["width"], image["height"])
            for image in instances.get("images", [])
        }

        for ann in instances.get("annotations", []):
            ann_id = ann.get("id", "<missing_id>")
            image_id = ann.get("image_id")
            bbox = ann.get("bbox")

            if image_id not in image_id_to_size:
                continue

            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            image_width, image_height = image_id_to_size[image_id]

            if x < 0 or y < 0:
                errors.append(f"[{split}] Annotation {ann_id}: negative bbox coordinates={bbox}")

            if x + w > image_width:
                errors.append(
                    f"[{split}] Annotation {ann_id}: bbox exceeds image width. "
                    f"bbox={bbox}, image_width={image_width}"
                )

            if y + h > image_height:
                errors.append(
                    f"[{split}] Annotation {ann_id}: bbox exceeds image height. "
                    f"bbox={bbox}, image_height={image_height}"
                )

    return errors

def validate_dataset(data_dir: Path | None = None) -> dict:
    data_dir = data_dir or get_default_data_dir()

    report = {
        "structure_errors": validate_structure(data_dir),
        "image_errors": validate_images(data_dir),
        "annotation_errors": validate_annotations(data_dir),
        "category_errors": validate_categories(data_dir),
        "bbox_errors": validate_bbox_bounds(data_dir),
    }

    report["passed"] = all(len(errors) == 0 for errors in report.values())

    return report