import argparse
import json
from pathlib import Path

import pandas as pd


def load_coco_annotations(data_dir: Path, split: str) -> pd.DataFrame:
    ann_path = data_dir / "annotations" / f"instances_{split}2017.json"

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {
        cat["id"]: cat["name"]
        for cat in coco["categories"]
    }

    rows = []

    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]

        rows.append(
            {
                "annotation_id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "category_name": categories[ann["category_id"]],
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": w * h,
                "aspect_ratio": w / h if h > 0 else None,
                "iscrowd": ann.get("iscrowd", 0),
            }
        )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    print("\n========== Object count per class ==========")
    print(
        df.groupby(["category_id", "category_name"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .to_string(index=False)
    )

    print("\n========== BBox area stats ==========")
    print(
        df.groupby(["category_id", "category_name"])["area"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\n========== BBox width stats ==========")
    print(
        df.groupby(["category_id", "category_name"])["width"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\n========== BBox height stats ==========")
    print(
        df.groupby(["category_id", "category_name"])["height"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\n========== Aspect ratio stats ==========")
    print(
        df.groupby(["category_id", "category_name"])["aspect_ratio"]
        .describe()
        .round(2)
        .to_string()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = load_coco_annotations(data_dir, args.split)

    summarize(df)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()