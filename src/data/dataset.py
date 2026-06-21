from torch.utils.data import Dataset
from src.data.coco import get_split_paths, get_default_data_dir
from pathlib import Path
import json
import torch
from PIL import Image

class CarDDDataset(Dataset):
    def __init__(self, data_dir, split, transforms=None):

        self.data_dir = Path(data_dir) if data_dir is not None else get_default_data_dir()
        paths = get_split_paths(self.data_dir, split)
        path = Path(paths['annotations_path'])
        self.transforms = transforms
        self.samples = []

        self.annotations = {}

        with open(path, 'r') as f:
            instances = json.load(f)
            for annotation in instances['annotations']:
                image_id = annotation['image_id']
                if image_id not in self.annotations:
                    self.annotations[image_id] = {
                        'boxes': [],
                        'labels': [],
                        'areas': [],
                        'iscrowd': [],
                    }

                x, y, w, h = annotation['bbox']
                box = [x, y, x + w, y + h]
                self.annotations[image_id]['boxes'].append(box)
                self.annotations[image_id]['labels'].append(annotation['category_id'])
                self.annotations[image_id]['areas'].append(annotation['area'])
                self.annotations[image_id]['iscrowd'].append(annotation.get('iscrowd', 0))

            for image in instances['images']:
                annotations = self.annotations.get(
                    image['id'],
                    {'boxes': [], 'labels': [], 'areas': [], 'iscrowd': []},
                )
                target = {
                    "image_id": torch.tensor([image["id"]], dtype=torch.int64),
                    "boxes": torch.tensor(annotations['boxes'], dtype=torch.float32).reshape(-1, 4),
                    "labels": torch.tensor(annotations['labels'], dtype=torch.int64),
                    "area": torch.tensor(annotations['areas'], dtype=torch.float32),
                    "iscrowd": torch.tensor(annotations['iscrowd'], dtype=torch.int64),
                    "path": Path(paths['images_dir'] / image['file_name']),
                    "width": image['width'],
                    "height": image['height']
                }
                self.samples.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target = self.samples[idx]
        image, target = Image.open(target['path']).convert('RGB'), target
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target
