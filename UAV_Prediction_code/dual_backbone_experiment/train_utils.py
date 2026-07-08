from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def _compute_position_error(outputs, targets, radar_xy=(200.0, 200.0), r_max=400.0):
    """
    outputs / targets: [B, 2K] or [B, 4]
    将归一化坐标还原到真实坐标后，计算每个样本的平均 UAV 位置误差
    """
    radar = np.array(radar_xy, dtype=np.float32).reshape(1, 1, 2)

    out_np = outputs.detach().cpu().numpy()
    tgt_np = targets.detach().cpu().numpy()

    out_real = out_np.reshape(out_np.shape[0], -1, 2) * r_max + radar
    tgt_real = tgt_np.reshape(tgt_np.shape[0], -1, 2) * r_max + radar

    uav_err = np.linalg.norm(out_real - tgt_real, axis=2)   # [B, num_uav]
    mean_uav_err = uav_err.mean(axis=1)                     # [B]
    return mean_uav_err.sum(), out_real.shape[0]


def evaluate_dual_input(model, device, dataloader, criterion, radar_xy=(200.0, 200.0), r_max=400.0):
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        total_pos_err = 0.0
        n_samples_total = 0

        for inputs1, _, targets in dataloader:
            inputs1 = inputs1.to(device)
            targets = targets.to(device)

            outputs = model(inputs1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            pos_err_sum, n_batch = _compute_position_error(outputs, targets, radar_xy=radar_xy, r_max=r_max)
            total_pos_err += pos_err_sum
            n_samples_total += n_batch

        mean_loss = total_loss / len(dataloader)
        mean_pos_err = total_pos_err / n_samples_total
        return mean_loss, mean_pos_err


def evaluate_single_input(model, device, dataloader, criterion, radar_xy=(200.0, 200.0), r_max=400.0):
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        total_pos_err = 0.0
        n_samples_total = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            pos_err_sum, n_batch = _compute_position_error(outputs, targets, radar_xy=radar_xy, r_max=r_max)
            total_pos_err += pos_err_sum
            n_samples_total += n_batch

        mean_loss = total_loss / len(dataloader)
        mean_pos_err = total_pos_err / n_samples_total
        return mean_loss, mean_pos_err


def save_summary(save_dir, summary_dict):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)


def train_dual_input(
    model,
    teacher,
    regression_task,
    device,
    save_dir,
    lambda_feat=10.0,
    n_epochs=120,
    lr=3e-4,
    test_every=5,
    weight_decay=0.0,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(teacher.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    writer = SummaryWriter(log_dir=str(save_dir / "tb"))

    best_loss = float("inf")
    best_epoch = -1
    best_pos_err = float("inf")

    for epoch in range(n_epochs):
        model.train()
        teacher.train()

        running_loss = 0.0
        running_reg = 0.0
        running_feat = 0.0

        for i, (inputs1, inputs2, targets) in enumerate(regression_task.trainloader):
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            preds, feat1 = model(inputs1, return_features=True)
            feat2 = teacher(inputs2)

            loss_reg = criterion(preds, targets)
            loss_feat = 1.0 - F.cosine_similarity(feat1, feat2, dim=1).mean()
            loss = loss_reg + lambda_feat * loss_feat

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_reg += loss_reg.item()
            running_feat += loss_feat.item()

            step = epoch * len(regression_task.trainloader) + i
            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/Loss_Reg", loss_reg.item(), step)
            writer.add_scalar("Train/Loss_Feat", loss_feat.item(), step)

        scheduler.step()

        avg_train_loss = running_loss / len(regression_task.trainloader)
        avg_reg_loss = running_reg / len(regression_task.trainloader)
        avg_feat_loss = running_feat / len(regression_task.trainloader)

        print(
            f"[Train][Dual] Epoch {epoch + 1}/{n_epochs} | "
            f"Loss={avg_train_loss:.6f} | Reg={avg_reg_loss:.6f} | Feat={avg_feat_loss:.6f}"
        )

        if (epoch + 1) % test_every == 0:
            test_loss, pos_err = evaluate_dual_input(
                model, device, regression_task.testloader, criterion
            )
            print(f"[Eval][Dual] Epoch {epoch + 1}: TestLoss={test_loss:.6f}, PositionError={pos_err:.3f}")

            writer.add_scalar("Eval/TestLoss", test_loss, epoch + 1)
            writer.add_scalar("Eval/PositionError", pos_err, epoch + 1)

            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch + 1
                best_pos_err = pos_err
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                torch.save(teacher.state_dict(), save_dir / "best_teacher.pth")
                print(f"✅ Best dual model saved at epoch {best_epoch}")

    writer.close()

    save_summary(
        save_dir,
        {
            "mode": "dual",
            "best_epoch": best_epoch,
            "best_norm_mse": best_loss,
            "best_position_error": best_pos_err,
            "lambda_feat": lambda_feat,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    )
    return best_epoch, best_loss


def train_single_input(
    model,
    regression_task,
    device,
    save_dir,
    n_epochs=120,
    lr=3e-4,
    test_every=5,
    weight_decay=0.0,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    writer = SummaryWriter(log_dir=str(save_dir / "tb"))

    best_loss = float("inf")
    best_epoch = -1
    best_pos_err = float("inf")

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(regression_task.trainloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            step = epoch * len(regression_task.trainloader) + i
            writer.add_scalar("Train/Loss", loss.item(), step)

        scheduler.step()

        avg_train_loss = running_loss / len(regression_task.trainloader)
        print(f"[Train][Single] Epoch {epoch + 1}/{n_epochs} | Loss={avg_train_loss:.6f}")

        if (epoch + 1) % test_every == 0:
            test_loss, pos_err = evaluate_single_input(
                model, device, regression_task.testloader, criterion
            )
            print(f"[Eval][Single] Epoch {epoch + 1}: TestLoss={test_loss:.6f}, PositionError={pos_err:.3f}")

            writer.add_scalar("Eval/TestLoss", test_loss, epoch + 1)
            writer.add_scalar("Eval/PositionError", pos_err, epoch + 1)

            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch + 1
                best_pos_err = pos_err
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                print(f"✅ Best single model saved at epoch {best_epoch}")

    writer.close()

    save_summary(
        save_dir,
        {
            "mode": "single",
            "best_epoch": best_epoch,
            "best_norm_mse": best_loss,
            "best_position_error": best_pos_err,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    )
    return best_epoch, best_loss