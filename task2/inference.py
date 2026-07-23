import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from kornia.feature import LoFTR


TILE_SIZE    = 512
TILE_OVERLAP = 64


def load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def pad_to_multiple(img: np.ndarray, multiple: int = 8):
    h, w = img.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    padded = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
    return padded, (h, w)


def to_tensor(gray: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(gray.astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0).to(device)


def match_single(model: LoFTR,
                 gray_a: np.ndarray,
                 gray_b: np.ndarray,
                 device: torch.device):
    pa, (oh_a, ow_a) = pad_to_multiple(gray_a)
    pb, (oh_b, ow_b) = pad_to_multiple(gray_b)

    with torch.no_grad():
        out = model({"image0": to_tensor(pa, device),
                     "image1": to_tensor(pb, device)})

    kp0 = out["keypoints0"].cpu().numpy()
    kp1 = out["keypoints1"].cpu().numpy()
    cf  = out["confidence"].cpu().numpy()

    mask = (
        (kp0[:, 0] < ow_a) & (kp0[:, 1] < oh_a) &
        (kp1[:, 0] < ow_b) & (kp1[:, 1] < oh_b)
    )
    return kp0[mask], kp1[mask], cf[mask]


def match_tiled(model: LoFTR,
                gray_a: np.ndarray,
                gray_b: np.ndarray,
                device: torch.device,
                tile_size: int = TILE_SIZE,
                overlap: int = TILE_OVERLAP):
    H, W  = gray_a.shape
    stride = tile_size - overlap
    margin = overlap // 2

    all_kp0, all_kp1, all_cf = [], [], []

    for row in range(0, H - margin, stride):
        for col in range(0, W - margin, stride):
            r0, r1 = row, min(row + tile_size, H)
            c0, c1 = col, min(col + tile_size, W)

            ta = gray_a[r0:r1, c0:c1]
            tb = gray_b[r0:r1, c0:c1]

            if ta.shape[0] < 8 or ta.shape[1] < 8:
                continue

            kp0, kp1, cf = match_single(model, ta, tb, device)

            th, tw = ta.shape
            inner = (
                (kp0[:, 0] > margin) & (kp0[:, 0] < tw - margin) &
                (kp0[:, 1] > margin) & (kp0[:, 1] < th - margin)
            )
            kp0, kp1, cf = kp0[inner], kp1[inner], cf[inner]

            kp0[:, 0] += c0;  kp0[:, 1] += r0
            kp1[:, 0] += c0;  kp1[:, 1] += r0

            all_kp0.append(kp0)
            all_kp1.append(kp1)
            all_cf.append(cf)

    if not all_kp0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)

    return (
        np.concatenate(all_kp0),
        np.concatenate(all_kp1),
        np.concatenate(all_cf),
    )


GREEN_IN  = (0, 220, 0)
RED_OUT   = (0, 0, 220)
DOT_IN    = (0, 255, 0)
DOT_OUT   = (0, 80, 255)


def ransac_split(kpts0: np.ndarray,
                 kpts1: np.ndarray,
                 conf: np.ndarray,
                 reproj_thresh: float = 4.0):
    if len(kpts0) < 4:
        return kpts0, kpts1, conf, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)

    _, mask = cv2.findHomography(
        kpts0.astype(np.float32),
        kpts1.astype(np.float32),
        cv2.RANSAC,
        reproj_thresh,
    )
    if mask is None:
        return kpts0, kpts1, conf, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)

    inliers = mask.ravel().astype(bool)
    return (kpts0[inliers],  kpts1[inliers],  conf[inliers],
            kpts0[~inliers], kpts1[~inliers], conf[~inliers])


def draw_matches(rgb_a: np.ndarray,
                 rgb_b: np.ndarray,
                 kpts0: np.ndarray,
                 kpts1: np.ndarray,
                 conf: np.ndarray,
                 max_draw: int = 200,
                 label_a: str = "Image A",
                 label_b: str = "Image B") -> np.ndarray:
    H   = max(rgb_a.shape[0], rgb_b.shape[0])
    off = rgb_a.shape[1]
    canvas = np.zeros((H, off + rgb_b.shape[1], 3), dtype=np.uint8)
    canvas[: rgb_a.shape[0], :off]  = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2BGR)
    canvas[: rgb_b.shape[0],  off:] = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2BGR)
    canvas[:, off] = 255

    if len(kpts0) >= 4:
        kp0_in, kp1_in, cf_in, kp0_out, kp1_out, _ = ransac_split(kpts0, kpts1, conf)
    else:
        kp0_in, kp1_in, cf_in = kpts0, kpts1, conf
        kp0_out = kp1_out = np.zeros((0, 2))

    n_in_show  = max_draw * 4 // 5
    n_out_show = max_draw - n_in_show

    if len(cf_in) > 0:
        order = np.argsort(cf_in)[::-1][:n_in_show]
        kp0_in, kp1_in = kp0_in[order], kp1_in[order]
    kp0_out = kp0_out[:n_out_show]
    kp1_out = kp1_out[:n_out_show]

    for (x0, y0), (x1, y1) in zip(kp0_out, kp1_out):
        cv2.line(canvas, (int(x0), int(y0)), (int(x1) + off, int(y1)),
                 RED_OUT, 1, cv2.LINE_AA)
    for (x0, y0), (x1, y1) in zip(kp0_out, kp1_out):
        cv2.circle(canvas, (int(x0),       int(y0)), 3, DOT_OUT, -1)
        cv2.circle(canvas, (int(x1) + off, int(y1)), 3, DOT_OUT, -1)

    for (x0, y0), (x1, y1) in zip(kp0_in, kp1_in):
        cv2.line(canvas, (int(x0), int(y0)), (int(x1) + off, int(y1)),
                 GREEN_IN, 1, cv2.LINE_AA)
    for (x0, y0), (x1, y1) in zip(kp0_in, kp1_in):
        cv2.circle(canvas, (int(x0),       int(y0)), 4, DOT_IN, -1)
        cv2.circle(canvas, (int(x1) + off, int(y1)), 4, DOT_IN, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, label_a, (10, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, label_b, (off + 10, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    n_in  = len(kp0_in)
    n_tot = n_in + len(kp0_out)
    ratio = n_in / max(n_tot, 1)
    info  = (f"Inliers (green): {n_in}   Outliers (red): {n_tot - n_in}"
             f"   Ratio: {ratio:.0%}")
    cv2.putText(canvas, info, (10, H - 10), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    return canvas


def run_pair(model: LoFTR,
             path_a: str,
             path_b: str,
             device: torch.device,
             conf_thresh: float = 0.3,
             output_path: str | None = None,
             save_coords: str | None = None,
             max_draw: int = 200,
             label_a: str = "Image A",
             label_b: str = "Image B"):
    gray_a = load_gray(path_a)
    gray_b = load_gray(path_b)
    rgb_a  = load_rgb(path_a)
    rgb_b  = load_rgb(path_b)

    large = max(gray_a.shape[0], gray_a.shape[1],
                gray_b.shape[0], gray_b.shape[1]) > TILE_SIZE

    if large:
        print("  Large image — using tiled matching")
        kpts0, kpts1, conf = match_tiled(model, gray_a, gray_b, device)
    else:
        kpts0, kpts1, conf = match_single(model, gray_a, gray_b, device)

    mask  = conf >= conf_thresh
    kpts0 = kpts0[mask]
    kpts1 = kpts1[mask]
    conf  = conf[mask]

    n   = len(kpts0)
    med = float(np.median(conf)) if n > 0 else 0.0
    print(f"  {n} matches  |  median conf {med:.3f}")

    if output_path:
        os.makedirs(Path(output_path).parent, exist_ok=True)
        vis = draw_matches(rgb_a, rgb_b, kpts0, kpts1, conf,
                           max_draw=max_draw, label_a=label_a, label_b=label_b)
        cv2.imwrite(output_path, vis)
        print(f"  Saved visualisation → {output_path}")

    if save_coords:
        os.makedirs(Path(save_coords).parent, exist_ok=True)
        data = {
            "image_a":    str(path_a),
            "image_b":    str(path_b),
            "n_matches":  n,
            "keypoints_a": kpts0.tolist(),
            "keypoints_b": kpts1.tolist(),
            "confidence":  conf.tolist(),
        }
        with open(save_coords, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"  Saved coordinates  → {save_coords}")

    return kpts0, kpts1, conf


def run_batch(args: argparse.Namespace, model: LoFTR, device: torch.device) -> None:
    with open(args.metadata) as fh:
        meta = json.load(fh)

    os.makedirs(args.output_dir, exist_ok=True)

    for m in meta:
        pid    = m["pair_id"]
        path_a = str(Path(args.data_dir) / m["image_a"])
        path_b = str(Path(args.data_dir) / m["image_b"])
        print(f"\n[pair {pid:03d}]  {m['date_a']} ↔ {m['date_b']}")

        out_vis    = str(Path(args.output_dir) / f"pair_{pid:03d}_matches.png")
        out_coords = str(Path(args.output_dir) / f"pair_{pid:03d}_coords.json")

        run_pair(
            model, path_a, path_b, device,
            conf_thresh=args.conf_thresh,
            output_path=out_vis,
            save_coords=out_coords,
            max_draw=args.max_draw,
            label_a=m["date_a"],
            label_b=m["date_b"],
        )


def build_model(weights: str | None, device: torch.device) -> LoFTR:
    model = LoFTR(pretrained="outdoor").to(device)
    if weights and Path(weights).exists():
        ckpt = torch.load(weights, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded fine-tuned weights from {weights}")
    else:
        print("Using pre-trained LoFTR (outdoor weights)")
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoFTR inference on Sentinel-2 image pairs"
    )
    p.add_argument("--img-a",  default=None)
    p.add_argument("--img-b",  default=None)
    p.add_argument("--output", default="results/matches.png")
    p.add_argument("--save-coords", default=None)
    p.add_argument("--batch",      action="store_true")
    p.add_argument("--data-dir",   default="dataset_pairs/pairs")
    p.add_argument("--metadata",   default="dataset_pairs/pairs_metadata.json")
    p.add_argument("--output-dir", default="results/")
    p.add_argument("--weights",     default=None)
    p.add_argument("--conf-thresh", type=float, default=0.3)
    p.add_argument("--max-draw",    type=int,   default=60)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    model = build_model(args.weights, device)

    if args.batch:
        run_batch(args, model, device)
    elif args.img_a and args.img_b:
        print(f"\n{args.img_a}  ↔  {args.img_b}")
        run_pair(
            model, args.img_a, args.img_b, device,
            conf_thresh=args.conf_thresh,
            output_path=args.output,
            save_coords=args.save_coords,
            max_draw=args.max_draw,
        )
    else:
        print("Provide --img-a and --img-b for single-pair mode, or --batch for all pairs.")
        print("Run with --help for full usage.")
