
import csv
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler
from torchvision import transforms

from datasets.dataset import build_dataset


MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

def evaluation_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class ImageManifest(Dataset):
    def __init__(self, csv_path, split, raw=False, celeba_val_fraction=None, seed=42, celeba=False, data_root=None):
        path = Path(csv_path).resolve()
        self.root = Path(data_root) if data_root is not None else Path(__file__).resolve().parent

        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            rows = list(reader)

        self.rows = [row for row in rows if row["split"] == split]

        if celeba_val_fraction is not None:
            test_ids = {row["subject_id"] for row in rows if row["split"] == "test"}
            candidates = sorted({row["subject_id"] for row in self.rows} - test_ids)
            selected = set(np.random.default_rng(seed).choice(
                candidates, size=round(len(candidates) * celeba_val_fraction), replace=False
            ).tolist())
            self.rows = [row for row in self.rows if row["subject_id"] in selected]
            self.selected_subjects = sorted(selected)

        self.celeba = celeba
        self.video_ids = [row["video_id"] for row in self.rows] if "video_id" in fields else None
        self.tf = evaluation_transform()
        self.raw = raw

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["path"])

        if not path.is_absolute():
            path = self.root / path

        if self.celeba:
            image = celeba_crop(path)
        else:
            with Image.open(path) as opened:
                image = opened.convert("RGB")

        return (image if self.raw else self.tf(image)), int(row["label"]), "external"


def celeba_crop(path):
    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    box = path.with_name(path.stem + "_BB.txt")
    values = box.read_text().split()

    x, y, w, h = (int(float(v)) for v in values[:4])
    x, y = int(x * width / 224), int(y * height / 224)
    w, h = int(w * width / 224), int(h * height / 224)

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    crop = image[y1:y2, x1:x2]

    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def domain_class_sampler(training, ocim_domains, seed, batch_size, frames_per_video=2):
    base = training.base
    cells = Counter()

    for domain, dataset in zip(ocim_domains, base.datasets[:3]):
        for _, label in dataset.samples:
            cells[(domain, int(label))] += 1

    celeba = base.datasets[3]
    for row in celeba.rows:
        cells[("CelebA", int(row["label"]))] += 1

    weights = []
    for domain, dataset in zip(ocim_domains, base.datasets[:3]):
        weights.extend(1 / (8 * cells[(domain, int(label))]) for _, label in dataset.samples)
    weights.extend(1 / (8 * cells[("CelebA", int(row["label"]))]) for row in celeba.rows)

    count = max(sum(len(ds) for ds in base.datasets[:3]), batch_size) * frames_per_video
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), count, replacement=True, generator=generator)
    return sampler, count


class PairedViews(Dataset):
    def __init__(self, base):
        self.base = base
        self.geometry = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip()])
        self.style = transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.10, hue=.02)
        self.tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label, domain = self.base[index]
        image = self.geometry(image)
        return self.tensor(image), self.tensor(self.style(image)), label, domain


def make_dataset(domains, split, seed, manifests, celeba_val_fraction=None, data_root=None, celeba_root=None):
    datasets = []

    for domain in domains:
        if domain in ("O", "C", "I", "M"):
            datasets.append(build_dataset(
                domain, split, None if split == "train" else evaluation_transform(), data_root=data_root
            ))
        else:
            datasets.append(ImageManifest(
                manifests[domain], split, raw=split == "train",
                celeba_val_fraction=celeba_val_fraction if domain == "CelebA" else None,
                seed=seed, celeba=domain == "CelebA", data_root=celeba_root
            ))

    base = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return PairedViews(base) if split == "train" else base


class ProtocolManifest(ImageManifest):
    def __init__(self, manifest, split, data_root=None):
        super().__init__(manifest, split, raw=split == "train", data_root=data_root)
        videos = {}

        for row in self.rows:
            videos.setdefault(row["video_id"], []).append(row)

        self.rows = [sorted(frames, key=lambda row: row["path"]) for frames in videos.values()]
        self.video_ids = list(videos)

    def __getitem__(self, index):
        frames = self.rows[index]
        row = frames[torch.randint(len(frames), (1,)).item()] if self.raw else frames[len(frames) // 2]
        path = Path(row["path"])

        if not path.is_absolute():
            path = self.root / path

        with Image.open(path) as opened:
            image = opened.convert("RGB")

        return (image if self.raw else self.tf(image)), int(row["label"]), "external"


def make_protocol_dataset(manifest, split, data_root=None):
    base = ProtocolManifest(manifest, split, data_root=data_root)
    return PairedViews(base) if split == "train" else base
