import numpy as np
import torch
import albumentations as A


def get_train_transforms(cfg):
    aug_cfg = cfg.get("augmentations", {}).get("train", {})

    transforms = []

    if aug_cfg.get("horizontal_flip", False):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug_cfg.get("brightness_contrast", False):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            )
        )

    if aug_cfg.get("motion_blur", False):
        transforms.append(A.MotionBlur(blur_limit=5, p=0.2))

    if aug_cfg.get("gauss_noise", False):
        transforms.append(A.GaussNoise(p=0.2))

    if aug_cfg.get("affine", False):
        transforms.append(
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-7, 7),
                p=0.3,
            )
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.2,
        ),
    )


def get_valid_transforms(cfg=None):
    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0,
        ),
    )


class DetectionAlbumentations:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        image = np.array(image)

        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()

        transformed = self.transform(
            image=image,
            bboxes=boxes,
            labels=labels,
        )

        image = transformed["image"]
        boxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.array(transformed["labels"], dtype=np.int64)

        h, w = image.shape[:2]

        if len(boxes) > 0:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]

            keep = (box_w > 1.0) & (box_h > 1.0)

            boxes = boxes[keep]
            labels = labels[keep]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        return {
            "image": image,
            "target": {
                "image_id": target["image_id"],
                "boxes": boxes,
                "labels": labels,
                "area": area,
                "iscrowd": iscrowd,
                "path": target["path"],
                "width": target["width"],
                "height": target["height"],
            },
        }["image"], {
            "image_id": target["image_id"],
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "path": target["path"],
            "width": target["width"],
            "height": target["height"],
        }