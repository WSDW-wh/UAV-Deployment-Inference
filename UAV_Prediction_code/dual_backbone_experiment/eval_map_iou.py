import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from data_load_ablation import RegressionTaskDataDual
from models_dual_backbone import SingleBackboneRegressor, DualBackboneRegressor
from JAM_model import Jam2img_array


def tensor_to_rgb_uint8(x):
    """
    x: torch tensor, shape [3, H, W], value range [0, 1]
    return: uint8 RGB image, shape [H, W, 3]
    """
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def rgb_to_effect_mask(rgb, white_thr=250):
    """
    将态势图转成二值有效区域。
    你的图是白底，蓝色区域表示有效态势区域。
    因此只要不是接近白色，就认为是有效区域。
    """
    return np.any(rgb < white_thr, axis=-1)


def compute_iou(mask_pred, mask_ref, eps=1e-6):
    inter = np.logical_and(mask_pred, mask_ref).sum()
    union = np.logical_or(mask_pred, mask_ref).sum()

    if union == 0:
        return 1.0

    return inter / (union + eps)


def denormalize_positions(pred, radar_xy=(200.0, 200.0), r_max=400.0):
    """
    pred: [B, 4], normalized positions
    return: [B, 2, 2], real positions
    """
    radar = np.array(radar_xy, dtype=np.float32).reshape(1, 1, 2)
    pred_np = pred.detach().cpu().numpy()
    pred_real = pred_np.reshape(pred_np.shape[0], -1, 2) * r_max + radar
    return pred_real


def compute_position_error(pred, target, radar_xy=(200.0, 200.0), r_max=400.0):
    radar = np.array(radar_xy, dtype=np.float32).reshape(1, 1, 2)

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    pred_real = pred_np.reshape(pred_np.shape[0], -1, 2) * r_max + radar
    target_real = target_np.reshape(target_np.shape[0], -1, 2) * r_max + radar

    err = np.linalg.norm(pred_real - target_real, axis=2)  # [B, num_uav]
    return err.mean(axis=1)  # [B]


def build_model(backbone, use_attention, use_dual, output_dim, pretrained=False):
    if use_dual:
        model = DualBackboneRegressor(
            backbone_type=backbone,
            use_attention=use_attention,
            pretrained=pretrained,
            out_dim=output_dim,
            embed_dim=256,
        )
    else:
        model = SingleBackboneRegressor(
            backbone_type=backbone,
            use_attention=use_attention,
            pretrained=pretrained,
            out_dim=output_dim,
            embed_dim=256,
        )

    return model


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    _, _, sample_target = sample_batch
    output_dim = sample_target.shape[1]

    model = build_model(
        backbone=args.backbone,
        use_attention=args.use_attention,
        use_dual=args.use_dual,
        output_dim=output_dim,
        pretrained=args.pretrained,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.MSELoss()

    all_iou = []
    all_pos_err = []
    all_mse = []

    with torch.no_grad():
        for inputs1, inputs2, targets in tqdm(data.testloader):
            inputs1 = inputs1.to(device)
            targets = targets.to(device)

            preds = model(inputs1)

            loss = criterion(preds, targets)
            all_mse.append(loss.item())

            pos_err = compute_position_error(
                preds,
                targets,
                radar_xy=(args.radar_x, args.radar_y),
                r_max=args.r_max,
            )
            all_pos_err.extend(pos_err.tolist())

            pred_real = denormalize_positions(
                preds,
                radar_xy=(args.radar_x, args.radar_y),
                r_max=args.r_max,
            )

            batch_size = pred_real.shape[0]

            for i in range(batch_size):
                uav1 = pred_real[i, 0]
                uav2 = pred_real[i, 1]

                pred_img = Jam2img_array(
                    uav1,
                    uav2,
                    threshold=args.jsr_threshold,
                    image_size=400,
                    radar=(args.radar_x, args.radar_y),
                    radius=args.radius,
                )

                pred_img = Image.fromarray(pred_img).resize(
                    (args.resize_size, args.resize_size),
                    resample=Image.BILINEAR,
                )
                pred_rgb = np.asarray(pred_img)

                # 默认和 reference map 比，也就是 paired data 里的第二张 image
                if args.compare_to == "reference":
                    ref_rgb = tensor_to_rgb_uint8(inputs2[i])
                else:
                    ref_rgb = tensor_to_rgb_uint8(inputs1[i].detach().cpu())

                pred_mask = rgb_to_effect_mask(pred_rgb, white_thr=args.white_thr)
                ref_mask = rgb_to_effect_mask(ref_rgb, white_thr=args.white_thr)

                iou = compute_iou(pred_mask, ref_mask)
                all_iou.append(iou)

    result = {
        "checkpoint": args.checkpoint,
        "backbone": args.backbone,
        "use_attention": args.use_attention,
        "use_dual": args.use_dual,
        "compare_to": args.compare_to,
        "norm_mse": float(np.mean(all_mse)),
        "position_error": float(np.mean(all_pos_err)),
        "map_iou": float(np.mean(all_iou)),
        "map_iou_std": float(np.std(all_iou)),
        "success_rate_iou_0.7": float(np.mean(np.array(all_iou) >= 0.7)),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved result to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "plaincnn"])
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--use_dual", action="store_true")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--data_dir", type=str, default="../JAM_data")
    parser.add_argument("--train_csv", type=str, default="re_train_paired.csv")
    parser.add_argument("--test_csv", type=str, default="re_test_paired.csv")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--radar_x", type=float, default=200.0)
    parser.add_argument("--radar_y", type=float, default=200.0)
    parser.add_argument("--r_max", type=float, default=400.0)
    parser.add_argument("--radius", type=float, default=200.0)

    parser.add_argument("--jsr_threshold", type=float, default=0.02)
    parser.add_argument("--white_thr", type=int, default=250)

    parser.add_argument(
        "--compare_to",
        type=str,
        default="reference",
        choices=["reference", "desired"],
        help="reference: compare with mechanism-generated map; desired: compare with transformed input map",
    )

    parser.add_argument("--output", type=str, default="map_iou_result.json")

    args = parser.parse_args()
    evaluate(args)