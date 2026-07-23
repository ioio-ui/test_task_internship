import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kornia.feature import LoFTR


def sample_homography(h: int, w: int,
                      max_angle: float = 20.0,
                      max_shift: float = 0.15,
                      max_scale: float = 0.15) -> np.ndarray:
    angle = random.uniform(-max_angle, max_angle)
    scale = random.uniform(1 - max_scale, 1 + max_scale)
    tx = random.uniform(-max_shift * w, max_shift * w)
    ty = random.uniform(-max_shift * h, max_shift * h)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    H = np.eye(3, dtype=np.float32)
    H[:2] = M
    return H


class SatellitePairDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 metadata_path: str,
                 patch_size: int = 256,
                 n_per_epoch: int = 500):
        self.patch_size = patch_size
        self.n_per_epoch = n_per_epoch

        with open(metadata_path) as fh:
            meta = json.load(fh)

        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for m in meta:
            pa = Path(images_dir) / m["image_a"]
            pb = Path(images_dir) / m["image_b"]
            img_a = np.array(Image.open(pa).convert("L"), dtype=np.float32) / 255.0
            img_b = np.array(Image.open(pb).convert("L"), dtype=np.float32) / 255.0
            self.pairs.append((img_a, img_b))

        assert self.pairs, "No pairs loaded — check --data-dir and --metadata paths."

    def __len__(self) -> int:
        return self.n_per_epoch

    def __getitem__(self, _):
        img_a, img_b = random.choice(self.pairs)
        H_h, W_w = img_a.shape
        sz = self.patch_size

        r = random.randint(0, H_h - sz)
        c = random.randint(0, W_w - sz)
        patch_a = img_a[r: r + sz, c: c + sz].copy()
        patch_b = img_b[r: r + sz, c: c + sz].copy()

        H_mat = sample_homography(sz, sz, max_angle=10.0, max_shift=0.08, max_scale=0.08)
        patch_b = cv2.warpPerspective(patch_b, H_mat, (sz, sz))

        t0 = torch.from_numpy(patch_a).unsqueeze(0)
        t1 = torch.from_numpy(patch_b).unsqueeze(0)
        H_t = torch.from_numpy(H_mat)

        return {"image0": t0, "image1": t1, "H": H_t}


def matching_loss(output: dict,
                  H_mat: torch.Tensor,
                  reproj_thresh: float = 4.0) -> torch.Tensor:
    mkpts0 = output["keypoints0"]
    mkpts1 = output["keypoints1"]
    conf   = output["confidence"]

    if mkpts0.shape[0] == 0:
        return torch.zeros(1, device=H_mat.device, requires_grad=True).squeeze()

    ones    = torch.ones(mkpts0.shape[0], 1, device=mkpts0.device)
    pts_h   = torch.cat([mkpts0.float(), ones], dim=1)
    proj    = (H_mat.float() @ pts_h.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3].clamp(min=1e-6)

    err = torch.norm(proj_xy - mkpts1.float(), dim=1)
    gt  = (err < reproj_thresh).float().detach()

    err_norm    = err / reproj_thresh
    reproj_loss = (conf.float() * err_norm.clamp(max=10.0)).mean()

    conf_bce = F.binary_cross_entropy(
        conf.float().clamp(1e-6, 1 - 1e-6), gt
    )

    return reproj_loss + conf_bce


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Data   : {args.data_dir}")
    print(f"Output : {args.output}")

    dataset = SatellitePairDataset(
        images_dir=args.data_dir,
        metadata_path=args.metadata,
        patch_size=args.patch_size,
        n_per_epoch=args.n_per_epoch,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = LoFTR(pretrained="outdoor").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(Path(args.output).parent, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch:>3}/{args.epochs}")

        for batch in pbar:
            img0 = batch["image0"].to(device)
            img1 = batch["image1"].to(device)
            H    = batch["H"].squeeze(0).to(device)

            optimizer.zero_grad()

            with torch.enable_grad():
                output = model({"image0": img0, "image1": img1})
            loss = matching_loss(output, H, reproj_thresh=args.reproj_thresh)

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"  avg loss: {avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "args": vars(args),
                },
                args.output,
            )
            print(f"  Checkpoint saved -> {args.output}  (loss={best_loss:.5f})")

    print(f"\nTraining complete. Best loss: {best_loss:.5f}")
    print(f"Weights saved to  : {args.output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune LoFTR on Sentinel-2 cross-season pairs"
    )
    p.add_argument("--data-dir",       default="dataset_pairs/pairs")
    p.add_argument("--metadata",       default="dataset_pairs/pairs_metadata.json")
    p.add_argument("--output",         default="checkpoints/loftr_satellite.pth")
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--patch-size",     type=int,   default=256)
    p.add_argument("--n-per-epoch",    type=int,   default=500)
    p.add_argument("--reproj-thresh",  type=float, default=4.0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
