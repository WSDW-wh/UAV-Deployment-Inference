import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_load_ablation import RegressionTaskDataDual
from models_dual_backbone import DualBackboneRegressor, SingleBackboneRegressor


_GRID_CACHE = {}


def tensor_to_rgb_uint8(x):
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def rgb_to_mask(rgb, white_thr=250):
    """White background is invalid area; non-white pixels are situation area."""
    return np.any(rgb < white_thr, axis=-1)


def mask_metrics(pred_mask, ref_mask, eps=1e-8):
    pred_mask = pred_mask.astype(bool)
    ref_mask = ref_mask.astype(bool)

    inter = np.logical_and(pred_mask, ref_mask).sum()
    union = np.logical_or(pred_mask, ref_mask).sum()
    pred_area = pred_mask.sum()
    ref_area = ref_mask.sum()

    iou = 1.0 if union == 0 else inter / (union + eps)
    precision = 1.0 if pred_area == 0 else inter / (pred_area + eps)
    recall = 1.0 if ref_area == 0 else inter / (ref_area + eps)
    dice = 1.0 if pred_area + ref_area == 0 else 2.0 * inter / (pred_area + ref_area + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pred_area": int(pred_area),
        "ref_area": int(ref_area),
        "intersection": int(inter),
        "union": int(union),
    }


def denormalize_positions(x, radar_xy=(200.0, 200.0), r_max=400.0):
    radar = np.array(radar_xy, dtype=np.float32).reshape(1, 1, 2)
    x_np = x.detach().cpu().numpy()
    return x_np.reshape(x_np.shape[0], -1, 2) * float(r_max) + radar


def position_errors(preds, targets, radar_xy=(200.0, 200.0), r_max=400.0):
    pred_real = denormalize_positions(preds, radar_xy, r_max)
    target_real = denormalize_positions(targets, radar_xy, r_max)
    per_uav = np.linalg.norm(pred_real - target_real, axis=2)
    return per_uav.mean(axis=1), per_uav


def build_model(args, output_dim):
    cls = DualBackboneRegressor if args.use_dual else SingleBackboneRegressor
    return cls(
        backbone_type=args.backbone,
        use_attention=args.use_attention,
        pretrained=args.pretrained,
        out_dim=output_dim,
        embed_dim=args.embed_dim,
    )


def load_checkpoint(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def get_mechanism_grid(image_size, radar, radius):
    key = (int(image_size), float(radar[0]), float(radar[1]), float(radius))
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    h = w = int(image_size)
    x_img, y_img = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    circle_mask = ((x_img - radar[0]) ** 2 + (y_img - radar[1]) ** 2) <= float(radius) ** 2

    # Match JAM_model.Jam2img_array: x is image x, y is converted to math coordinates.
    target_x = x_img[circle_mask]
    target_y = (h - 1) - y_img[circle_mask]
    targets = np.stack([target_x, target_y], axis=1).astype(np.float32)

    _GRID_CACHE[key] = (circle_mask, targets)
    return _GRID_CACHE[key]


def vectorized_jamm_power(radar, targets, uav):
    yita = 0.313
    theta5 = 0.0175
    g_j = 5.0
    p_t = 1.0
    g_r = 40.0

    radar = np.asarray(radar, dtype=np.float32)
    uav = np.asarray(uav, dtype=np.float32)
    vector1 = targets - radar[None, :]
    vector2 = uav - radar

    mag1 = np.linalg.norm(vector1, axis=1)
    mag2 = np.linalg.norm(vector2)
    denom = np.maximum(mag1 * mag2, 1e-12)
    cos_theta = np.sum(vector1 * vector2[None, :], axis=1) / denom
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    g_r1 = np.empty_like(theta, dtype=np.float32)
    main_lobe = (theta >= 0.0) & (theta < theta5 / 2.0)
    side_lobe = (theta >= theta5 / 2.0) & (theta < np.pi / 2.0)
    back_lobe = ~np.logical_or(main_lobe, side_lobe)

    g_r1[main_lobe] = g_r
    theta_safe = np.maximum(theta[side_lobe], 1e-12)
    g_r1[side_lobe] = yita * (theta5 / theta_safe) ** 2 * g_r
    g_r1[back_lobe] = yita * (2.0 * theta5 / np.pi) ** 2 * g_r

    radar_uav_dist = max(float(np.linalg.norm(radar - uav)), 1e-12)
    return p_t * g_j * g_r1 / (4.0 * np.pi) ** 2 / radar_uav_dist**2


def vectorized_echo_power(radar, targets):
    delta = 25.0
    p_ts = 10.0
    g_r = 40.0

    radar = np.asarray(radar, dtype=np.float32)
    r_h = np.linalg.norm(targets - radar[None, :], axis=1)
    r_h = np.maximum(r_h, 1e-12)
    q_h = p_ts * g_r / (4.0 * np.pi) / r_h**2
    return q_h * g_r * delta / (4.0 * np.pi) ** 2 / r_h**2


def fast_jam2img_array(
    uav1,
    uav2,
    threshold=0.02,
    image_size=400,
    radar=(200.0, 200.0),
    radius=200.0,
    deep_color=(0, 0, 255),
):
    h = w = int(image_size)
    radar_arr = np.asarray(radar, dtype=np.float32)
    circle_mask, targets = get_mechanism_grid(h, radar_arr, radius)

    pj = vectorized_jamm_power(radar_arr, targets, uav1) + vectorized_jamm_power(radar_arr, targets, uav2)
    pr = vectorized_echo_power(radar_arr, targets)
    jsr = pj / np.maximum(pr, 1e-12)
    active = jsr < float(threshold)

    out = np.ones((h, w, 3), dtype=np.float32) * 255.0
    if np.any(active):
        severity = 1.0 - jsr[active] / float(threshold)
        severity = np.clip(severity, 0.0, 1.0).reshape(-1, 1)
        deep = np.asarray(deep_color, dtype=np.float32).reshape(1, 3)
        colors = (1.0 - severity) * 255.0 + severity * deep

        ys, xs = np.where(circle_mask)
        out[ys[active], xs[active], :] = colors

    return out.astype(np.uint8)


def make_mechanism_rgb(uav_xy, args):
    img = fast_jam2img_array(
        uav_xy[0],
        uav_xy[1],
        threshold=args.jsr_threshold,
        image_size=args.mechanism_size,
        radar=(args.radar_x, args.radar_y),
        radius=args.radius,
    )
    if args.mechanism_size != args.resize_size:
        img = Image.fromarray(img).resize(
            (args.resize_size, args.resize_size),
            resample=Image.BILINEAR,
        )
        img = np.asarray(img)
    return img


def save_example(out_dir, index, input_rgb, pred_rgb, ref_rgb, pred_mask, ref_mask):
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap = np.ones_like(input_rgb) * 255
    overlap[np.logical_and(ref_mask, ~pred_mask)] = np.array([220, 40, 40], dtype=np.uint8)
    overlap[np.logical_and(pred_mask, ~ref_mask)] = np.array([40, 120, 230], dtype=np.uint8)
    overlap[np.logical_and(pred_mask, ref_mask)] = np.array([40, 180, 80], dtype=np.uint8)

    panels = [input_rgb, pred_rgb, ref_rgb, overlap.astype(np.uint8)]
    canvas = np.concatenate(panels, axis=1)
    Image.fromarray(canvas).save(out_dir / f"sample_{index:05d}.png")


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def evaluate_one(args, write_outputs=True, desc="Evaluating"):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data = RegressionTaskDataDual(
        grayscale=False,
        image_folder_path=Path(args.data_dir),
        resize_size=args.resize_size,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        radar_xy=(args.radar_x, args.radar_y),
        r_max=args.r_max,
        normalize_target=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    sample_batch = next(iter(data.testloader))
    output_dim = sample_batch[2].shape[1]

    model = build_model(args, output_dim).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    criterion = nn.MSELoss(reduction="sum")
    n_elements = 0
    mse_sum = 0.0
    rows = []
    ious = []
    dices = []
    precisions = []
    recalls = []
    pos_errs = []
    global_index = 0

    with torch.no_grad():
        for inputs1, inputs2, targets in tqdm(data.testloader, desc=desc):
            if args.max_samples > 0:
                remaining = args.max_samples - global_index
                if remaining <= 0:
                    break
                inputs1 = inputs1[:remaining]
                inputs2 = inputs2[:remaining]
                targets = targets[:remaining]

            inputs1 = inputs1.to(device)
            targets = targets.to(device)
            preds = model(inputs1)

            mse_sum += criterion(preds, targets).item()
            n_elements += targets.numel()

            batch_pos_errs, per_uav_errs = position_errors(
                preds,
                targets,
                radar_xy=(args.radar_x, args.radar_y),
                r_max=args.r_max,
            )
            pos_errs.extend(batch_pos_errs.tolist())

            pred_real = denormalize_positions(preds, (args.radar_x, args.radar_y), args.r_max)
            target_real = denormalize_positions(targets, (args.radar_x, args.radar_y), args.r_max)

            for i in range(pred_real.shape[0]):
                input_rgb = tensor_to_rgb_uint8(inputs1[i])
                pred_rgb = make_mechanism_rgb(pred_real[i], args)

                if args.compare_to == "input":
                    ref_rgb = input_rgb
                elif args.compare_to == "reference":
                    ref_rgb = tensor_to_rgb_uint8(inputs2[i])
                elif args.compare_to == "target_mechanism":
                    ref_rgb = make_mechanism_rgb(target_real[i], args)
                else:
                    raise ValueError(f"Unknown compare_to: {args.compare_to}")

                pred_mask = rgb_to_mask(pred_rgb, white_thr=args.white_thr)
                ref_mask = rgb_to_mask(ref_rgb, white_thr=args.white_thr)
                m = mask_metrics(pred_mask, ref_mask)

                ious.append(m["iou"])
                dices.append(m["dice"])
                precisions.append(m["precision"])
                recalls.append(m["recall"])

                row = {
                    "index": global_index,
                    "iou": m["iou"],
                    "dice": m["dice"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "pred_area": m["pred_area"],
                    "ref_area": m["ref_area"],
                    "position_error_mean": float(batch_pos_errs[i]),
                }
                for k in range(per_uav_errs.shape[1]):
                    row[f"position_error_uav{k + 1}"] = float(per_uav_errs[i, k])
                rows.append(row)

                if args.save_examples > 0 and global_index < args.save_examples:
                    save_example(
                        Path(args.examples_dir),
                        global_index,
                        input_rgb,
                        pred_rgb,
                        ref_rgb,
                        pred_mask,
                        ref_mask,
                    )

                global_index += 1

    iou_arr = np.asarray(ious, dtype=np.float64)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "test_csv": args.test_csv,
        "num_samples": int(len(rows)),
        "device": str(device),
        "backbone": args.backbone,
        "use_attention": bool(args.use_attention),
        "use_dual": bool(args.use_dual),
        "compare_to": args.compare_to,
        "white_thr": int(args.white_thr),
        "jsr_threshold": float(args.jsr_threshold),
        "norm_mse": float(mse_sum / max(n_elements, 1)),
        "position_error": summarize(pos_errs),
        "map_iou": summarize(ious),
        "map_dice": summarize(dices),
        "map_precision": summarize(precisions),
        "map_recall": summarize(recalls),
        "success_rate_iou_0.50": float(np.mean(iou_arr >= 0.50)) if iou_arr.size else None,
        "success_rate_iou_0.70": float(np.mean(iou_arr >= 0.70)) if iou_arr.size else None,
        "success_rate_iou_0.90": float(np.mean(iou_arr >= 0.90)) if iou_arr.size else None,
    }

    if write_outputs:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if args.per_sample_csv:
            csv_path = Path(args.per_sample_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Saved summary to: {output_path}")
        if args.per_sample_csv:
            print(f"Saved per-sample metrics to: {args.per_sample_csv}")
        if args.save_examples > 0:
            print(f"Saved examples to: {args.examples_dir}")

    return result, rows


def infer_model_config(checkpoint, default_args):
    run_name = checkpoint.parent.name.lower()
    backbone = "plaincnn" if "plaincnn" in run_name else "resnet18" if "resnet18" in run_name else default_args.backbone
    use_dual = "dual" in run_name
    use_attention = ("attn" in run_name) or ("attention" in run_name)
    return backbone, use_dual, use_attention


def flatten_summary(result):
    return {
        "run": Path(result["checkpoint"]).parent.name,
        "checkpoint": result["checkpoint"],
        "backbone": result["backbone"],
        "use_dual": result["use_dual"],
        "use_attention": result["use_attention"],
        "compare_to": result["compare_to"],
        "num_samples": result["num_samples"],
        "norm_mse": result["norm_mse"],
        "position_error_mean": result["position_error"]["mean"],
        "position_error_std": result["position_error"]["std"],
        "map_iou_mean": result["map_iou"]["mean"],
        "map_iou_std": result["map_iou"]["std"],
        "map_dice_mean": result["map_dice"]["mean"],
        "map_precision_mean": result["map_precision"]["mean"],
        "map_recall_mean": result["map_recall"]["mean"],
        "success_rate_iou_0.50": result["success_rate_iou_0.50"],
        "success_rate_iou_0.70": result["success_rate_iou_0.70"],
        "success_rate_iou_0.90": result["success_rate_iou_0.90"],
    }


def write_summary_csv(path, results):
    if not results:
        return
    rows = [flatten_summary(r) for r in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_all(args):
    checkpoints = sorted(Path(args.models_root).glob("*/best_model.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No best_model.pth found under: {args.models_root}")

    all_results = []
    failures = []

    print(f"Found {len(checkpoints)} checkpoints under {args.models_root}")
    for checkpoint in checkpoints:
        model_args = copy.copy(args)
        model_args.checkpoint = str(checkpoint)
        model_args.backbone, model_args.use_dual, model_args.use_attention = infer_model_config(checkpoint, args)

        run_dir = checkpoint.parent
        suffix = args.compare_to
        model_args.output = str(run_dir / f"situation_iou_{suffix}.json")
        if args.per_sample_csv:
            model_args.per_sample_csv = str(run_dir / f"situation_iou_{suffix}_per_sample.csv")
        model_args.examples_dir = str(run_dir / f"iou_examples_{suffix}")

        print(
            f"\n[{run_dir.name}] backbone={model_args.backbone}, "
            f"use_dual={model_args.use_dual}, use_attention={model_args.use_attention}"
        )
        try:
            result, _ = evaluate_one(model_args, write_outputs=True, desc=f"Evaluating {run_dir.name}")
            all_results.append(result)
        except Exception as exc:
            failures.append({"run": run_dir.name, "checkpoint": str(checkpoint), "error": repr(exc)})
            print(f"Failed {run_dir.name}: {exc}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = {"results": all_results, "failures": failures}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    csv_path = output_path.with_suffix(".csv")
    write_summary_csv(csv_path, all_results)

    print("\nAll-model summary:")
    for r in all_results:
        print(
            f"{Path(r['checkpoint']).parent.name:24s} "
            f"IoU={r['map_iou']['mean']:.4f} "
            f"PosErr={r['position_error']['mean']:.3f} "
            f"MSE={r['norm_mse']:.6f}"
        )
    if failures:
        print(f"\nFailures: {len(failures)}. See {output_path}")
    print(f"Saved aggregate JSON to: {output_path}")
    print(f"Saved aggregate CSV to: {csv_path}")


def evaluate(args):
    if args.checkpoint:
        evaluate_one(args, write_outputs=True)
    else:
        evaluate_all(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate IoU between input situation maps and mechanism maps reconstructed from predicted UAV positions."
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--models_root", type=str, default=str(THIS_DIR / "runs_exp"))
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "plaincnn"])
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--use_dual", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=256)

    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "JAM_data"))
    parser.add_argument("--train_csv", type=str, default="re_train_paired.csv")
    parser.add_argument("--test_csv", type=str, default="re_test_paired.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means evaluate all samples.")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--radar_x", type=float, default=200.0)
    parser.add_argument("--radar_y", type=float, default=200.0)
    parser.add_argument("--r_max", type=float, default=400.0)
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--mechanism_size", type=int, default=400)
    parser.add_argument("--jsr_threshold", type=float, default=0.02)
    parser.add_argument("--white_thr", type=int, default=250)
    parser.add_argument(
        "--compare_to",
        type=str,
        default="input",
        choices=["input", "reference", "target_mechanism"],
        help="input: compare with re_image; reference: compare with image; target_mechanism: compare with map generated from ground-truth UAV positions.",
    )

    parser.add_argument("--output", type=str, default=str(THIS_DIR / "situation_iou_all_results.json"))
    parser.add_argument("--per_sample_csv", type=str, default="")
    parser.add_argument("--save_examples", type=int, default=0)
    parser.add_argument("--examples_dir", type=str, default=str(THIS_DIR / "iou_examples"))
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
