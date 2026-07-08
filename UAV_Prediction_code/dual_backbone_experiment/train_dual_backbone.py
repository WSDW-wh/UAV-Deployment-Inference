import argparse
import torch

from data_load_ablation import RegressionTaskDataDual, RegressionTaskDataSingle
from train_utils import train_dual_input, train_single_input
from models_dual_backbone import (
    SingleBackboneRegressor,
    DualBackboneRegressor,
    DualFeatureTeacher,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "plaincnn"])
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--use_dual", action="store_true")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_feat", type=float, default=10.0)
    parser.add_argument("--test_every", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="runs_dual_backbone/default")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================
    # 数据
    # =========================================================
    if args.use_dual:
        data = RegressionTaskDataDual(
            grayscale=False,
            resize_size=args.resize_size,
            batch_size=args.batch_size,
        )
        sample_batch = next(iter(data.trainloader))
        _, _, sample_target = sample_batch
        output_dim = sample_target.shape[1]
    else:
        data = RegressionTaskDataSingle(
            grayscale=False,
            resize_size=args.resize_size,
            batch_size=args.batch_size,
            use_column=0,   # 单输入默认用 re_image
        )
        sample_batch = next(iter(data.trainloader))
        _, sample_target = sample_batch
        output_dim = sample_target.shape[1]

    # =========================================================
    # 模型
    # =========================================================
    if args.use_dual:
        model = DualBackboneRegressor(
            backbone_type=args.backbone,
            use_attention=args.use_attention,
            pretrained=args.pretrained,
            out_dim=output_dim,
            embed_dim=256,
        ).to(device)

        teacher = DualFeatureTeacher(
            backbone_type=args.backbone,
            use_attention=args.use_attention,
            pretrained=args.pretrained,
            embed_dim=256,
        ).to(device)
    else:
        model = SingleBackboneRegressor(
            backbone_type=args.backbone,
            use_attention=args.use_attention,
            pretrained=args.pretrained,
            out_dim=output_dim,
            embed_dim=256,
        ).to(device)

        teacher = None

    print("=" * 60)
    print(f"Using backbone : {args.backbone}")
    print(f"Use attention  : {args.use_attention}")
    print(f"Use dual       : {args.use_dual}")
    print(f"Pretrained     : {args.pretrained}")
    print(f"Output dim     : {output_dim}")
    print(f"Save dir       : {args.save_dir}")
    print("=" * 60)

    # =========================================================
    # 训练
    # =========================================================
    if args.use_dual:
        train_dual_input(
            model=model,
            teacher=teacher,
            regression_task=data,
            device=device,
            save_dir=args.save_dir,
            lambda_feat=args.lambda_feat,
            n_epochs=args.epochs,
            lr=args.lr,
            test_every=args.test_every,
            weight_decay=args.weight_decay,
        )
    else:
        train_single_input(
            model=model,
            regression_task=data,
            device=device,
            save_dir=args.save_dir,
            n_epochs=args.epochs,
            lr=args.lr,
            test_every=args.test_every,
        )