
import argparse
import csv
import json
from pathlib import Path

import torch

from cape_jepa.metrics import metrics, youden
from cape_jepa.utils import score_dataset
from datasets.data_loader import make_dataset, make_protocol_dataset
from datasets.dataset import original_dataset
from models.model import build_model, load_sources
from models.sequential import SequentialPAD


def write_result_csv(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)


def load_checkpoint(project, path):
    create, _, _ = load_sources(project)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "student" in checkpoint:
        config = checkpoint.get("config", {})
        if config.get("stage") == "ce":
            model = create(pretrained=True)
            model.load_state_dict(checkpoint["student"])
        else:
            initial = checkpoint.get("initial_checkpoint", checkpoint.get("original_checkpoint"))
            if config.get("student_architecture") == "sequential_pad":
                model = SequentialPAD()
                model.load_state_dict(checkpoint["student"])
            else:
                model, _ = build_model(project, initial)
                model.student.load_state_dict(checkpoint["student"])
                model = model.student
    else:
        model = create(pretrained=True)
        model.load_state_dict(checkpoint)

    return model


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--workflow", required=True, choices=("two_stage", "cape_finetune"))
    selected, remaining = selector.parse_known_args(argv)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)

    if selected.workflow == "two_stage":
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--target", required=True)
        p.add_argument("--celeba-manifest", type=Path)
        p.add_argument("--celeba-root", type=Path)
        p.add_argument("--protocol-manifest", type=Path)
        p.add_argument("--protocol-root", type=Path)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--test-threshold", action="store_true")
    else:
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--test-threshold", action="store_true")

    return selected.workflow, p.parse_args(remaining)


def evaluate_main(args):
    project = Path(__file__).resolve().parent.parent
    manifests = {"CelebA": args.celeba_manifest}
    fraction = 0.1 if args.target == "CelebA" and args.protocol_manifest is None else None

    if args.protocol_manifest is not None:
        validation = make_protocol_dataset(args.protocol_manifest, "val", data_root=args.protocol_root)
        test = make_protocol_dataset(args.protocol_manifest, "test", data_root=args.protocol_root)
    else:
        validation = make_dataset([args.target], "val", args.seed, manifests,
                                  celeba_val_fraction=fraction, data_root=args.data_root, celeba_root=args.celeba_root)
        test = make_dataset([args.target], "test", args.seed, manifests, data_root=args.data_root, celeba_root=args.celeba_root)

    model = load_checkpoint(project, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    vy, vs = score_dataset(model, validation, args.batch_size, args.workers, device, "manifest")
    ty, ts = score_dataset(model, test, args.batch_size, args.workers, device, "manifest")
    threshold = youden(ty, ts) if args.test_threshold else youden(vy, vs)
    metric_values = metrics(ty, ts, threshold)

    result = {
        "target": args.target, "checkpoint": str(args.checkpoint), "threshold": threshold,
        "threshold_source": "test" if args.test_threshold else "target_validation",
        "validation_count": len(vy), "test_count": len(ty),
        **{key: metric_values[key] for key in ("APCER", "BPCER", "HTER", "AUC")}
    }

    if fraction is not None:
        result["validation_subjects"] = validation.selected_subjects
        result["validation_fraction_subjects"] = fraction
        result["seed"] = args.seed

    write_result_csv(args.output, result)
    print(json.dumps(result, indent=2))


def evaluate_sequential(args):
    full_checkpoint = Path(args.checkpoint).name in {"checkpoint_best.pt", "checkpoint_last.pt"}
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=not full_checkpoint)
    config = ck["args"]

    device = torch.device(args.device)
    model = SequentialPAD(config["dim"], config["lora_rank"], config["lora_alpha"]).to(device)
    model.load_state_dict(ck["student"])
    records, _ = score_dataset(model, original_dataset(config["target"], "test", data_root=args.data_root),
                               args.batch_size, args.workers, device, "ocim")

    y = [row["label"] for row in records]
    scores = [row["score"] for row in records]
    threshold = youden(y, scores) if args.test_threshold else ck["threshold"]
    result = metrics(y, scores, threshold)
    result.update({
        "target": config["target"], "mode": "control" if config.get("control") else config["mode"],
        "seed": config["seed"], "target_validation_selected_epoch": ck["epoch"],
        "test_used_for_selection": args.test_threshold,
        "threshold_source": "test" if args.test_threshold else "target_validation"
    })

    write_result_csv(args.output, result)
    score_path = Path(args.output).with_suffix(".scores.csv")
    with score_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(result)


def main():
    workflow, args = parse_args()
    if workflow == "two_stage":
        evaluate_main(args)
    else:
        evaluate_sequential(args)


if __name__ == "__main__":
    main()
