
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.data_loader import ImageManifest


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def rng_state(generator):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler": generator.get_state()
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    generator.set_state(state["sampler"].cpu())


@torch.no_grad()
def score_dataset(model, dataset, batch_size, workers, device, mode):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda" if mode == "ocim" else False)
    model.eval()

    if mode == "ocim":
        records, total_ce = [], 0.
        for images, labels, domains in loader:
            logits = model(images.to(device)).float()
            total_ce += torch.nn.functional.cross_entropy(logits, labels.to(device), reduction="sum").item()
            scores = logits.softmax(1)[:, 1].cpu().tolist()
            predictions = logits.argmax(1).cpu().tolist()
            records.extend(
                {"label": int(y), "score": float(s), "prediction": int(p), "dataset": d}
                for y, s, p, d in zip(labels.tolist(), scores, predictions, domains)
            )

        return records, total_ce / len(records)

    labels, values = [], []
    for images, y, _ in loader:
        prediction = model(images.to(device)).float().softmax(1)[:, 1]
        labels.extend(y.tolist())
        values.extend(prediction.cpu().tolist())

    if isinstance(dataset, ImageManifest) and dataset.video_ids is not None:
        grouped = defaultdict(list)
        for y, score, video in zip(labels, values, dataset.video_ids):
            grouped[video].append((y, score))

        labels, values = [], []
        for video, records in grouped.items():
            labels.append(records[0][0])
            values.append(float(np.mean([score for _, score in records])))

    return np.asarray(labels), np.asarray(values)
