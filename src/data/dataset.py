from torch.utils.data import Dataset
from src.data.coco import get_split_paths, get_default_data_dir
from pathlib import Path
import json
import torch
from PIL import Image
from collections import defaultdict

class CarDDDataset(Dataset):
    def __init__(self, data_dir, split, transforms=None):

        self.data_dir = data_dir or get_default_data_dir()
        paths = get_split_paths(self.data_dir, split)
        path = Path(paths['annotations_path'])
        self.transforms = transforms
        self.samples = []

        self.annotations = defaultdict(lambda: defaultdict(list))

        with open(path, 'r') as f:
            instances = json.load(f)
            for annotation in instances['annotations']:
                x, y, w, h = annotation['bbox']
                box = [x, y, x + w, y + h]
                self.annotations[annotation['image_id']]['boxes'].append(box)
                self.annotations[annotation['image_id']]['labels'].append(annotation['category_id'])
                self.annotations[annotation['image_id']]['areas'].append(annotation['area'])
                self.annotations[annotation['image_id']]['iscrowd'].append(annotation['iscrowd'])

            for image in instances['images']:
                target = {
                    "image_id": torch.tensor([image["id"]], dtype=torch.int64),
                    "boxes": torch.tensor(self.annotations[image['id']]['boxes'], dtype=torch.float32),
                    "labels": torch.tensor(self.annotations[image['id']]['labels'], dtype=torch.int64),
                    "area": torch.tensor(self.annotations[image['id']]['areas'], dtype=torch.float32),
                    "iscrowd": torch.tensor(self.annotations[image['id']]['iscrowd'], dtype=torch.int64),
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