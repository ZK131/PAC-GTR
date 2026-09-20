"""Train the RGB TopoCrackNet model on the fixed L515 scene split.

The data directory is intentionally supplied by the user and is not included
in this repository.  It must contain data_train.npy, data_val.npy,
data_test.npy and their matching mask_*.npy arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from skimage.morphology import skeletonize
from torch import nn
from torch.utils.data import DataLoader, Dataset

from topocracknet import TopoCrackNet, loss_fn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class L515Dataset(Dataset):
    """L515 arrays with training-only geometric and photometric augmentation."""

    def __init__(self, root: Path, split: str, augment: bool) -> None:
        self.images = np.load(root / f"data_{split}.npy", mmap_mode="r")
        self.masks = np.load(root / f"mask_{split}.npy", mmap_mode="r")
        self.augment = augment
        self.skeletons = np.stack([skeletonize(mask > 0) for mask in self.masks]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index].copy().astype(np.float32) / 255.0
        region = (self.masks[index] > 0).astype(np.float32)
        centerline = self.skeletons[index].copy()
        if self.augment and random.random() < 0.5:
            image, region, centerline = image[:, ::-1].copy(), region[:, ::-1].copy(), centerline[:, ::-1].copy()
        if self.augment and random.random() < 0.35:
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-0.06, 0.06)
            image = np.clip(image * gain + bias, 0.0, 1.0)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        region_tensor = torch.from_numpy(region[None])
        center_tensor = torch.from_numpy(centerline[None])
        dilated = nn.functional.max_pool2d(region_tensor[None], 3, 1, 1)[0]
        eroded = -nn.functional.max_pool2d(-region_tensor[None], 3, 1, 1)[0]
        boundary_tensor = ((dilated - eroded) > 0.05).float()
        return image_tensor, region_tensor, center_tensor, boundary_tensor


@torch.no_grad()
def evaluate(model: TopoCrackNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in ("tp", "fp", "fn", "loss")}
    for batch in loader:
        image, region, center, boundary = [value.to(device, non_blocking=True) for value in batch]
        outputs = model(image)
        total_loss, _ = loss_fn(outputs, region, center, boundary)
        prediction = torch.sigmoid(outputs[0]) >= 0.5
        totals["tp"] += float((prediction & (region > 0.5)).sum())
        totals["fp"] += float((prediction & (region <= 0.5)).sum())
        totals["fn"] += float(((~prediction) & (region > 0.5)).sum())
        totals["loss"] += float(total_loss) * image.shape[0]
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1.0)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1.0)
    iou = totals["tp"] / max(totals["tp"] + totals["fp"] + totals["fn"], 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {"loss": totals["loss"] / len(loader.dataset), "iou": iou, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TopoCrackNet on a prepared L515 split.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory with data_*.npy and mask_*.npy arrays")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/topocracknet_l515"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    required = [args.data_root / f"{prefix}_{split}.npy" for prefix in ("data", "mask") for split in ("train", "val", "test")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing L515 arrays:\n" + "\n".join(missing))
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_set = L515Dataset(args.data_root, "train", augment=True)
    validation_set = L515Dataset(args.data_root, "val", augment=False)
    test_set = L515Dataset(args.data_root, "test", augment=False)
    loader_options = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)

    model = TopoCrackNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_f1, history = -1.0, []
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            image, region, center, boundary = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            total_loss, _ = loss_fn(model(image), region, center, boundary)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(total_loss.detach()) * image.shape[0]
        scheduler.step()
        validation = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss / len(train_set), "learning_rate": scheduler.get_last_lr()[0], **{f"val_{key}": value for key, value in validation.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["f1"] > best_f1:
            best_f1 = validation["f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation": validation, "seed": args.seed}, args.output_dir / "best.pt")

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, test_loader, device)
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "model": "TopoCrackNet",
        "protocol": "fixed L515 scene split",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["validation"],
        "test": test,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
