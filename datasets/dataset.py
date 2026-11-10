
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CROP_NAMES = {"O": "OULU", "C": "CASIA", "I": "RA", "M": "MSU"}


def load_samples(dataset, split, data_root):
    root = Path(data_root)
    membership = json.loads((Path(__file__).parent / "ocim_membership.json").read_text())
    folder = root / CROP_NAMES[dataset] / "full"

    return [
        (folder / ("real" if row["label"] == 1 else "attack") / row["video_id"], row["label"])
        for row in membership
        if row["dataset"] == dataset and row["split"] == split
    ]


class FacePADDataset(Dataset):
    def __init__(self, dataset, split, transform=None, data_root=None):
        self.dataset = dataset.upper()
        self.split = split
        self.transform = transform
        self.samples = load_samples(self.dataset, split, data_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, label = self.samples[index]
        image = self.load_preprocessed_frame(video)

        if self.transform:
            image = self.transform(image)

        return image, label, self.dataset

    def load_preprocessed_frame(self, video):
        frames = sorted(video.parent.glob(video.name + "_[0-9][0-9][0-9][0-9].png"))
        index = np.random.randint(len(frames)) if self.split == "train" else len(frames) // 2

        with Image.open(frames[index]) as image:
            return image.convert("RGB")


def build_dataset(dataset, split, transform=None, data_root=None):
    return FacePADDataset(dataset, split, transform, data_root)


def original_dataset(target, split, data_root=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return build_dataset(target, split, transform=transform, data_root=data_root)
