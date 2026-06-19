from pathlib import Path
import json
import pandas as pd

SPLIT_TO_DIR = {
    "train": "train2017",
    "val": "val2017",
    "test": "test2017",
}


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by walking up until data/raw/CarDD_COCO exists."""
    start = (start or Path.cwd()).resolve()

    for path in [start, *start.parents]:
        if (path / "data" / "raw" / "CarDD_COCO").exists():
            return path

    raise FileNotFoundError("Could not find data/raw/CarDD_COCO from current working directory")


def get_default_data_dir() -> Path:
    return find_project_root() / "data" / "raw" / "CarDD_COCO"


def read_coco_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_split_paths(data_dir: Path, split: str) -> dict[str, Path]:
    if split not in SPLIT_TO_DIR:
        raise ValueError(f"Unknown split: {split}. Expected one of {list(SPLIT_TO_DIR)}")

    annotations_dir = data_dir / "annotations"

    return {
        "images_dir": data_dir / SPLIT_TO_DIR[split],
        "annotations_path": annotations_dir / f"instances_{split}2017.json",
    }


def coco_to_dataframes(
    split: str,
    images_dir: Path,
    annotations_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coco = read_coco_json(annotations_path)

    images_df = pd.DataFrame(coco.get("images", []))
    annotations_df = pd.DataFrame(coco.get("annotations", []))
    categories_df = pd.DataFrame(coco.get("categories", []))

    if not images_df.empty:
        images_df = images_df.copy()
        images_df["split"] = split
        images_df["image_path"] = images_df["file_name"].apply(lambda name: images_dir / name)
        images_df["image_exists"] = images_df["image_path"].apply(lambda path: path.exists())

    if not annotations_df.empty:
        annotations_df = annotations_df.copy()
        annotations_df["split"] = split

        if "bbox" in annotations_df.columns:
            bbox = pd.DataFrame(
                annotations_df["bbox"].tolist(),
                columns=["bbox_x", "bbox_y", "bbox_width", "bbox_height"],
                index=annotations_df.index,
            )
            annotations_df = pd.concat([annotations_df, bbox], axis=1)

    if not categories_df.empty:
        categories_df = categories_df.copy()
        categories_df["split"] = split

    return images_df, annotations_df, categories_df


def load_cardd_split(
    split: str,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir or get_default_data_dir()
    paths = get_split_paths(data_dir, split)

    return coco_to_dataframes(
        split=split,
        images_dir=paths["images_dir"],
        annotations_path=paths["annotations_path"],
    )


def load_cardd_all_splits(
    data_dir: Path | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir or get_default_data_dir()

    split_frames = {}

    if verbose:
        print(f"Data dir: {data_dir}")

    for split in splits:
        paths = get_split_paths(data_dir, split)

        if verbose:
            print(
                f"{split:>5}: images_dir={paths['images_dir'].exists()}, "
                f"annotations={paths['annotations_path'].exists()}"
            )

        split_frames[split] = load_cardd_split(split, data_dir)

    images_df = pd.concat([frames[0] for frames in split_frames.values()], ignore_index=True)
    annotations_df = pd.concat([frames[1] for frames in split_frames.values()], ignore_index=True)

    categories_df = (
        pd.concat([frames[2] for frames in split_frames.values()], ignore_index=True)
        .drop(columns="split", errors="ignore")
        .drop_duplicates()
        .sort_values("id")
        .reset_index(drop=True)
    )

    if not annotations_df.empty and not categories_df.empty:
        category_id_to_name = categories_df.set_index("id")["name"].to_dict()
        annotations_df["category_name"] = annotations_df["category_id"].map(category_id_to_name)

    if verbose:
        print(f"Images:      {len(images_df):,}")
        print(f"Annotations: {len(annotations_df):,}")
        print(f"Categories:  {len(categories_df):,}")

    return images_df, annotations_df, categories_df