
import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from cape_jepa.metrics import metrics, youden
from cape_jepa.utils import restore_rng, rng_state, score_dataset, seed_everything, write_json
from datasets.data_loader import ImageManifest, domain_class_sampler, make_dataset, make_protocol_dataset, seed_worker
from datasets.dataset import original_dataset
from models.model import update_teacher
from models.sequential import SequentialPAD, SequentialTrainer, load_original


def validate(model, loader, device):
    model.eval()
    loss_sum, labels, scores = 0., [], []

    with torch.no_grad():
        for images, y, _ in loader:
            logits = model(images.to(device)).float()
            loss_sum += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
            labels.extend(y.tolist())
            scores.extend(logits.softmax(1)[:, 1].cpu().tolist())

    if isinstance(loader.dataset, ImageManifest) and loader.dataset.video_ids is not None:
        grouped = defaultdict(list)
        for label, score, video in zip(labels, scores, loader.dataset.video_ids):
            grouped[video].append((label, score))

        labels, scores = [], []
        for video, records in grouped.items():
            labels.append(records[0][0])
            scores.append(float(np.mean([score for _, score in records])))

        probability = np.clip(np.asarray(scores), 1e-7, 1 - 1e-7)
        y = np.asarray(labels)
        loss = float(np.mean(-y * np.log(probability) - (1 - y) * np.log1p(-probability)))
    else:
        loss = loss_sum / len(labels)

    y, scores = np.asarray(labels), np.asarray(scores)
    threshold = youden(y, scores)
    metric_values = metrics(y, scores, threshold)
    return {"loss": loss, "threshold": threshold, **{k: metric_values[k] for k in ("APCER", "BPCER", "HTER", "AUC")}}


def select_epoch(history, targets):
    """Select minimum validation CE, averaging equally across targets."""

    def score(row):
        return sum(float(row[f"{target}_loss"]) for target in targets) / len(targets)

    best = min(history, key=score)
    return int(best["epoch"]), score(best)


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--workflow", choices=("two_stage", "cape_finetune"), required=True)
    selected, remaining = selector.parse_known_args(argv)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)

    if selected.workflow == "cape_finetune":
        p.add_argument("--original-repo", default=str(Path(__file__).resolve().parent.parent / "OCIM"))
        p.add_argument("--original-checkpoint")
        p.add_argument("--control", action="store_true")
        p.add_argument("--target", required=True, choices=list("OCIM"))
        p.add_argument("--output", required=True)
        p.add_argument("--checkpoint-name")
        p.add_argument("--no-pretrained", action="store_true")
        p.add_argument("--lora-rank", type=int, default=8)
        p.add_argument("--lora-alpha", type=float, default=8)
        p.add_argument("--dim", type=int, default=192)
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--accum-steps", type=int, default=2)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--frames-per-video", type=int, default=2)
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--backbone-lr", type=float, default=2e-5)
        p.add_argument("--weight-decay", type=float, default=.01)
        p.add_argument("--grad-clip", type=float, default=1.)
        p.add_argument("--teacher-momentum", type=float, default=.99)
        p.add_argument("--mask-ratio", type=float, default=.3)
        p.add_argument("--prediction-weight", type=float, default=1.)
        p.add_argument("--masked-weight", type=float, default=.5)
        p.add_argument("--consistency-weight", type=float, default=.1)
        p.add_argument("--train-bn", action="store_true")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--no-amp", action="store_true")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--resume")

        args = p.parse_args(remaining)
        args.mode = "random"
        return selected.workflow, args

    p.add_argument("--sources", nargs="+", required=True, choices=["O", "C", "I", "M", "CelebA"])
    p.add_argument("--targets", nargs="+", required=True, choices=["O", "C", "I", "M", "CelebA"])
    p.add_argument("--stage", required=True, choices=["ce", "cape", "cape_no_prediction"])
    p.add_argument("--initial-checkpoint", type=Path)
    p.add_argument("--celeba-manifest", type=Path)
    p.add_argument("--celeba-root", type=Path)
    p.add_argument("--protocol-manifest", type=Path)
    p.add_argument("--protocol-root", type=Path)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--ocim-celeba-balanced", action="store_true")
    p.add_argument("--frames-per-video", type=int, default=2)
    p.add_argument("--mask-ratio", type=float, default=.3)
    p.add_argument("--prediction-weight", type=float, default=1.)
    p.add_argument("--masked-weight", type=float, default=.5)
    p.add_argument("--consistency-weight", type=float, default=.1)
    p.add_argument("--teacher-momentum", type=float, default=.99)

    args = p.parse_args(remaining)
    args.project = Path(__file__).resolve().parent.parent
    args.celeba_val_fraction = 0.1 if "CelebA" in args.targets else None
    return selected.workflow, args


def train_two_stage(args):
    manifests = {"CelebA": args.celeba_manifest}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.protocol_manifest is not None:
        training = make_protocol_dataset(args.protocol_manifest, "train", data_root=args.protocol_root)
        validations = {t: make_protocol_dataset(args.protocol_manifest, "val", data_root=args.protocol_root) for t in args.targets}
    else:
        training = make_dataset(args.sources, "train", args.seed, manifests, data_root=args.data_root, celeba_root=args.celeba_root)
        validations = {
            t: make_dataset([t], "val", args.seed, manifests,
                            celeba_val_fraction=args.celeba_val_fraction if t == "CelebA" else None,
                            data_root=args.data_root, celeba_root=args.celeba_root)
            for t in args.targets
        }

    if args.ocim_celeba_balanced:
        sampler, sample_count = domain_class_sampler(training, args.sources[:-1], args.seed, args.batch_size, args.frames_per_video)
    else:
        sampler, sample_count = None, len(training)

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=args.workers, generator=loader_generator)
    val_loaders = {
        t: DataLoader(ds, batch_size=64 if t == "CelebA" else args.batch_size, shuffle=False, num_workers=args.workers)
        for t, ds in validations.items()
    }

    if args.stage == "ce":
        from models.model import load_sources
        create_clip_lora, _, _ = load_sources(args.project)
        trainer = create_clip_lora(pretrained=True)
    else:
        student = load_original(SequentialPAD(), args.initial_checkpoint)
        trainer = SequentialTrainer(student)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.to(device)
    parameters = [p for p in trainer.parameters() if p.requires_grad]

    if args.stage == "ce":
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        groups = {}
        for name, param in trainer.named_parameters():
            if not param.requires_grad:
                continue
            backbone = "student.encoder." in name
            lora = "parametrizations." in name
            lr = args.backbone_lr if backbone and not lora else args.lr
            decay = 0. if param.ndim <= 1 or name.endswith("bias") else args.weight_decay
            groups.setdefault((lr, decay), []).append(param)

        optimizer = torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()])

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    if args.ocim_celeba_balanced:
        config["samples_per_epoch"] = sample_count
        config["sampling"] = "equal domain/class with replacement"
    if args.celeba_val_fraction is not None:
        config["celeba_validation_subjects"] = validations["CelebA"].selected_subjects

    config["initialization"] = "task_specific_ce" if args.initial_checkpoint else "pretrained_clip_lora"
    config["checkpoint_selection"] = ("minimum target-validation cross-entropy" if len(args.targets) == 1
                                      else "minimum mean target-validation cross-entropy")
    config["cape_activation"] = "from_first_update"
    if args.stage != "ce":
        config["student_architecture"] = "sequential_pad"

    config_path = args.output / "config.json"
    history_path = args.output / "metrics.csv"

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        trainer.load_state_dict(state["training_model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        for entry in optimizer.state.values():
            for key, value in entry.items():
                if torch.is_tensor(value):
                    entry[key] = value.to(device)

        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        loader_generator.set_state(state["loader_rng"])
        if sampler is not None:
            sampler.generator.set_state(state["sampler_rng"])

        with history_path.open(newline="") as f:
            history = list(csv.DictReader(f))

        completed = state["epoch"]
        history = history[:completed - 1] + [state["record"]]
        start = completed + 1
        with history_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(state["record"]))
            writer.writeheader()
            writer.writerows(history)
    else:
        args.output.mkdir(parents=True)
        (args.output / "checkpoints").mkdir()
        config_path.write_text(json.dumps(config, indent=2))
        history, start = [], 1

    best_stop_loss = min(
        (sum(float(row[f"{t}_loss"]) for t in args.targets) / len(args.targets)
         for row in history),
        default=float("inf")
    )
    epochs_without_improvement = 0
    for row in reversed(history):
        loss = sum(float(row[f"{t}_loss"]) for t in args.targets) / len(args.targets)
        if loss <= best_stop_loss:
            break
        epochs_without_improvement += 1

    for epoch in range(start, args.epochs + 1):
        trainer.train()
        total, count = 0., 0
        optimizer.zero_grad(set_to_none=True)

        for step, (weak, strong, labels, _) in enumerate(loader):
            weak, strong, labels = weak.to(device), strong.to(device), labels.to(device)
            prediction_weight = 0. if args.stage == "cape_no_prediction" else args.prediction_weight

            if args.stage == "ce":
                loss = nn.functional.cross_entropy(trainer(strong).float(), labels)
            else:
                losses = trainer.losses(weak, strong, labels, args.mask_ratio, True,
                                       prediction_weight, args.masked_weight, args.consistency_weight)
                loss = losses["total"]

            remaining = len(loader) - (step // args.accum_steps) * args.accum_steps
            (loss / min(args.accum_steps, remaining)).backward()
            if (step + 1) % args.accum_steps == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if args.stage != "ce":
                    update_teacher(trainer.teacher, trainer.student, args.teacher_momentum)

            total += loss.item() * len(labels)
            count += len(labels)

        measurements = {
            t: validate(trainer if args.stage == "ce" else trainer.student, v, device)
            for t, v in val_loaders.items()
        }
        record = {"epoch": epoch, "training_loss": total / count}
        for target, result in measurements.items():
            record.update({f"{target}_{key}": value for key, value in result.items()})

        state = {
            "epoch": epoch,
            "training_model": {k: v.detach().cpu().clone() for k, v in trainer.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "config": config,
            "validation": measurements,
            "record": record,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "loader_rng": loader_generator.get_state(),
            "sampler_rng": sampler.generator.get_state() if sampler is not None else None,
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        }

        student = (state["training_model"] if args.stage == "ce" else
                   {k[len("student."):]: v for k, v in state["training_model"].items() if k.startswith("student.")})

        common_loss = sum(float(record[f"{t}_loss"]) for t in args.targets) / len(args.targets)
        if not history or common_loss < min(
            sum(float(row[f"{t}_loss"]) for t in args.targets) / len(args.targets)
            for row in history
        ):
            torch.save({"student": student, "epoch": epoch}, args.output / "checkpoints" / "best_common.pt")

        for target in args.targets:
            if not history or float(record[f"{target}_loss"]) < min(float(row[f"{target}_loss"]) for row in history):
                torch.save({"student": student, "epoch": epoch}, args.output / "checkpoints" / f"best_{target}.pt")

        torch.save(state, args.output / "checkpoints" / "checkpoint_last.pt")

        history.append(record)
        with history_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(record), flush=True)

        if args.early_stopping_patience > 0:
            if common_loss < best_stop_loss:
                best_stop_loss = common_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    selected_epoch, selected_loss = select_epoch(history, args.targets)
    selection = ("minimum target-validation CE" if len(args.targets) == 1
                 else "minimum macro target-validation CE (equal domain weights)")
    selected_state = torch.load(args.output / "checkpoints" / "best_common.pt", map_location="cpu", weights_only=True)
    selected_student = selected_state["student"]
    checkpoint_name = "best_common_model.pth" if len(args.targets) > 1 else "best_model.pth"
    common_checkpoint = args.output / checkpoint_name

    torch.save({
        "student": selected_student,
        "initial_checkpoint": config["initial_checkpoint"],
        "selected_epoch": selected_epoch,
        "selection": selection,
        "selection_value": selected_loss,
        "selection_targets": list(args.targets),
        "config": config
    }, common_checkpoint)

    if args.stage == "ce":
        torch.save({
            "student": selected_student,
            "config": config,
            "selected_epoch": selected_epoch,
            "selection": selection,
            "selection_value": selected_loss,
            "selection_targets": list(args.targets),
            "role": "stage_transition"
        }, args.output / "stage_transition.pth")
    print(f"SELECTED COMMON: epoch {selected_epoch}, {selection}, loss {selected_loss:.8f}", flush=True)

    for target in args.targets:
        best = min(history, key=lambda r: float(r[f"{target}_loss"]))
        best_epoch = int(best["epoch"])
        state = torch.load(args.output / "checkpoints" / f"best_{target}.pt", map_location="cpu", weights_only=True)
        student = state["student"]

        if len(args.targets) > 1:
            output = args.output / "targets" / target / "best_model.pth"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "student": student,
                "initial_checkpoint": config["initial_checkpoint"],
                "selected_epoch": best_epoch,
                "selection_target": target,
                "selection": "minimum target-validation CE",
                "config": config
            }, output)
        print(f'BEST {target}: epoch {best_epoch}, validation loss {float(best[f"{target}_loss"]):.8f}', flush=True)


def train_ocim_finetune(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    checkpoint_name = args.checkpoint_name or (f"cape_{args.target}_control.pt" if args.control else f"cape_{args.target}.pt")
    output.mkdir(parents=True, exist_ok=True)

    sources = [d for d in "OCIM" if d != args.target]

    dataset = make_dataset(sources, "train", args.seed, {}, data_root=args.data_root)
    counts = Counter((d.dataset, label) for d in dataset.base.datasets for _, label in d.samples)
    weights = [1 / counts[(d.dataset, label)] for d in dataset.base.datasets for _, label in d.samples]
    generator = torch.Generator().manual_seed(args.seed)
    samples = max(len(dataset), args.batch_size) * args.frames_per_video
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), samples, replacement=True, generator=generator)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator)

    original = Path(args.original_checkpoint) if args.original_checkpoint else (
        Path(__file__).resolve().parent.parent / "checkpoints" / "main" / "clip_lora" / args.target / f"clip_lora_{args.target}.pth")
    student = load_original(SequentialPAD(args.dim, args.lora_rank, args.lora_alpha), original)
    model = SequentialTrainer(student, "random").to(device)
    if args.control:
        model.mode = "baseline"

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if args.control and name.startswith((
            "predictor.", "student.projection.", "student.norm.", "student.aux_head."
        )):
            continue
        lr = args.backbone_lr if name.startswith("student.encoder.") and "parametrizations." not in name else args.lr
        decay = 0 if param.ndim <= 1 or name.endswith("bias") else args.weight_decay
        groups.setdefault((lr, decay), []).append(param)

    optimizer = torch.optim.AdamW([
        {"params": ps, "lr": lr, "initial_lr": lr, "weight_decay": wd}
        for (lr, wd), ps in groups.items()
    ])
    amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16, init_scale=1024.)
    start, update, best_loss = 0, 0, float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["training_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
        scaler.load_state_dict(checkpoint["scaler"])
        start, update, best_loss = checkpoint["epoch"] + 1, checkpoint["update"], checkpoint["best_validation_loss"]
        restore_rng(checkpoint["rng"], generator)

    write_json({
        "args": vars(args),
        "train": {"sources": sources, "preprocessing": "offline MTCNN face crops"},
        "target_val": {"source": "preprocessed OCIM MTCNN face crops"},
        "torch": torch.__version__,
        "target_images_used_for_gradient_updates": 0,
        "checkpoint_selection": "target-validation mean cross-entropy loss (preprocessed MTCNN crops)",
        "threshold_selection": "target-validation Youden index",
        "original_checkpoint": str(original.resolve()),
        "training_variant": "CE continuation" if args.control else "CAPE-JEPA fine-tuning",
        "comparability_note": "Original checkpoint selected using target validation; second-stage source training uses preprocessed MTCNN crops and CAPE augmentation"
    }, output / "config.json")

    for epoch in range(start, args.epochs):
        model.train()
        if not args.train_bn and not args.no_pretrained:
            for module in model.student.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

        active = not args.control
        sums, count = {}, 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        for step, (weak, strong, labels, _) in enumerate(loader):
            weak, strong, labels = weak.to(device), strong.to(device), labels.to(device)
            window_start = (step // args.accum_steps) * args.accum_steps * args.batch_size
            window_samples = min(args.accum_steps * args.batch_size, samples - window_start)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                losses = model.losses(weak, strong, labels, args.mask_ratio, active,
                                      args.prediction_weight, args.masked_weight, args.consistency_weight)
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.) + float(value.detach()) * len(labels)
            count += len(labels)
            scaler.scale(losses["total"] * len(labels) / window_samples).backward()
            if (step + 1) % args.accum_steps == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               args.grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_teacher(model.teacher, model.student, args.teacher_momentum)
                update += 1

        records, validation_loss = score_dataset(
            model.student, original_dataset(args.target, "val", data_root=args.data_root),
            args.batch_size, args.workers, device, "ocim"
        )
        threshold = youden([r["label"] for r in records], [r["score"] for r in records])
        val = metrics([r["label"] for r in records], [r["score"] for r in records], threshold)
        improved = validation_loss < best_loss
        best_loss = min(best_loss, validation_loss)

        result = {
            "epoch": epoch,
            "mode": "control" if args.control else args.mode,
            "auxiliary_active": active,
            "train": {k: v / count for k, v in sums.items()},
            "target_validation": val,
            "target_validation_loss": validation_loss,
            "validation_threshold": threshold,
            "best_validation_loss": best_loss,
            "minutes": (time.perf_counter() - t0) / 60
        }
        checkpoint = {
            "schema": 1,
            "epoch": epoch,
            "args": vars(args),
            "student": model.student.state_dict(),
            "training_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "threshold": threshold,
            "target_validation": val,
            "best_validation_loss": best_loss,
            "update": update,
            "rng": rng_state(generator)
        }

        torch.save(checkpoint, output / "checkpoint_last.pt")
        if improved:
            torch.save(checkpoint, output / "checkpoint_best.pt")
            write_json(records, output / "best_target_val_scores.json")
            torch.save({"student": model.student.state_dict(), "args": vars(args),
                        "threshold": threshold, "epoch": epoch}, output / checkpoint_name)

        with (output / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, allow_nan=False) + "\n")
        print(json.dumps(result, allow_nan=False), flush=True)


def main():
    workflow, args = parse_args()
    if workflow == "two_stage":
        train_two_stage(args)
    else:
        train_ocim_finetune(args)


if __name__ == "__main__":
    main()
