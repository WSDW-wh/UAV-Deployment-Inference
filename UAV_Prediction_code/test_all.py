# ============================================================
# test_best_feat_model_with_dataset_denorm.py
#
# 增强版测试脚本：
# 1) 加载 best_feat_model.pth 在 testloader 上进行推理
# 2) 反归一化到真实物理空间 (米)
# 3) 自动计算并打印论文级评估指标：
#    - 坐标均方根误差 (RMSE)
#    - 坐标平均绝对误差 (MAE)
#    - UAV1 / UAV2 的物理直线距离误差 (Euclidean Distance Error)
#    - 95% 置信区间最大距离误差
# 4) 保存预测结果到 test_predictions.npy
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from data_load_double import RegressionTaskData


# =====================================================
# Efficient Additive Attention
# =====================================================
class EfficientAdditiveAttention(nn.Module):
    def __init__(self, dim, token_dim=64, num_heads=2):
        super().__init__()
        self.to_query = nn.Linear(dim, token_dim * num_heads)
        self.to_key = nn.Linear(dim, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale = token_dim ** -0.5
        self.proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, dim)

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        x_tokens = x.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        q = F.normalize(self.to_query(x_tokens), dim=-1)
        k = F.normalize(self.to_key(x_tokens), dim=-1)

        A = (q @ self.w_g) * self.scale  # [B, N, 1]
        A = F.normalize(A, dim=1)

        G = torch.sum(A * q, dim=1)  # [B, D]
        G = einops.repeat(G, 'b d -> b n d', n=k.shape[1])

        out = self.proj(G * k) + q
        out = self.final(out)  # [B, N, C]

        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return out + residual


# =====================================================
# DiffConv
# =====================================================
class DiffConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.v = nn.Conv2d(in_ch, out_ch, k, padding=p)
        self.c = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)
        self.h = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)
        self.vd = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)
        self.a = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)

    def forward(self, x):
        return self.v(x) + self.c(x) + self.h(x) + self.vd(x) + self.a(x)


# =====================================================
# 主网络（测试阶段无教师分支）
# =====================================================
class CNNRegression(nn.Module):
    def __init__(self, image_size=(3, 224, 224)):
        super().__init__()
        C = image_size[0]

        self.conv1 = DiffConv(C, 4)
        self.bn1 = nn.BatchNorm2d(4)
        self.attn1 = EfficientAdditiveAttention(4)

        self.conv2 = DiffConv(4, 16)
        self.bn2 = nn.BatchNorm2d(16)
        self.attn2 = EfficientAdditiveAttention(16)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        feat_dim = 16 * (image_size[1] // 4) * (image_size[2] // 4)
        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x, return_features=False):
        x = self.pool1(self.attn1(F.relu(self.bn1(self.conv1(x)))))
        x = self.attn2(F.relu(self.bn2(self.conv2(x))))
        features = self.pool2(x)

        y = features.flatten(1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        if return_features:
            return y, features
        return y


# =====================================================
# 反归一化： norm: (p - radar)/r_max  ->  raw: p = norm*r_max + radar
# =====================================================
def denorm_xy_pairs_torch(y_norm: torch.Tensor, radar_xy=(200.0, 200.0), r_max=400.0) -> torch.Tensor:
    if y_norm.ndim != 2 or (y_norm.shape[1] % 2 != 0):
        return y_norm
    radar = torch.tensor(radar_xy, dtype=y_norm.dtype, device=y_norm.device).view(1, 1, 2)
    pts = y_norm.view(y_norm.shape[0], -1, 2)  # [B, K, 2]
    pts = pts * float(r_max) + radar
    return pts.view(y_norm.shape[0], -1)


# =====================================================
# 核心评估与统计计算
# =====================================================
def eval_best_model(device,
                    image_size=(3, 224, 224),
                    weight_path="best_feat_model.pth",
                    print_samples=3,
                    save_pred_npy=True,
                    pred_save_path="test_predictions.npy",
                    radar_xy=(200.0, 200.0),
                    r_max=400.0):
    regression_task = RegressionTaskData(
        grayscale=(image_size[0] == 1),
        resize_size=image_size[1],
        radar_xy=radar_xy,
        r_max=r_max,
        normalize_target=True
    )

    model = CNNRegression(image_size).to(device)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"❌ 找不到权重文件: {weight_path}")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print(f"✅ Loaded weights: {weight_path}")

    preds_raw_all, gts_raw_all = [], []
    shown = 0

    print("\n[Inferencing on Test Set...]")
    with torch.no_grad():
        for inputs1, _, targets in regression_task.testloader:
            inputs1 = inputs1.to(device)
            targets = targets.to(device)

            preds = model(inputs1)

            # 直接转到真实物理空间 (米)
            preds_raw = denorm_xy_pairs_torch(preds, radar_xy=radar_xy, r_max=r_max).cpu().numpy()
            gts_raw = denorm_xy_pairs_torch(targets, radar_xy=radar_xy, r_max=r_max).cpu().numpy()

            preds_raw_all.append(preds_raw)
            gts_raw_all.append(gts_raw)

            # 打印少量样本检查
            if shown < print_samples:
                b = min(inputs1.shape[0], print_samples - shown)
                for j in range(b):
                    print(f"----- Sample {shown + 1} -----")
                    print(f"GT (Raw m):   {gts_raw[j]}")
                    print(f"Pred(Raw m):  {preds_raw[j]}")
                    shown += 1
                    if shown >= print_samples:
                        break

    # 拼接所有的预测和真值矩阵 (N, 4)
    preds_raw_all = np.concatenate(preds_raw_all, axis=0)
    gts_raw_all = np.concatenate(gts_raw_all, axis=0)

    # ---------------- 统计计算核心 ---------------- #
    # 1. 坐标维度基础误差 (N, 4)
    abs_err = np.abs(preds_raw_all - gts_raw_all)
    sqr_err = np.square(preds_raw_all - gts_raw_all)

    mae_coord = np.mean(abs_err)
    rmse_coord = np.sqrt(np.mean(sqr_err))

    # 2. 物理直线距离误差 (Euclidean Distance)
    # 将 (N, 4) 转换为 (N, 2, 2) 分别代表两架无人机的 (x, y)
    preds_pts = preds_raw_all.reshape(-1, 2, 2)
    gts_pts = gts_raw_all.reshape(-1, 2, 2)

    # 计算欧氏距离 (N, 2)
    euclidean_dists = np.linalg.norm(preds_pts - gts_pts, axis=2)

    dist_uav1 = euclidean_dists[:, 0]
    dist_uav2 = euclidean_dists[:, 1]

    mean_dist_uav1 = np.mean(dist_uav1)
    mean_dist_uav2 = np.mean(dist_uav2)
    mean_dist_overall = np.mean(euclidean_dists)

    # 3. 95% 置信区间 (用于写论文："95%的样本误差在 X 米以内")
    p95_dist = np.percentile(euclidean_dists, 95)

    # 打印最终统计表格
    print("\n================= 论文级统计报告 (Raw Space / Meter) =================")
    print(f"📝 总体坐标 MAE (Mean Absolute Error)     : {mae_coord:.4f} 米")
    print(f"📝 总体坐标 RMSE (Root Mean Squared Error): {rmse_coord:.4f} 米")
    print("-" * 68)
    print(f"🎯 UAV_1 平均物理直线误差距离             : {mean_dist_uav1:.4f} 米")
    print(f"🎯 UAV_2 平均物理直线误差距离             : {mean_dist_uav2:.4f} 米")
    print(f"🎯 全局平均物理直线误差距离 (Mean Dist)   : {mean_dist_overall:.4f} 米")
    print(f"📊 95% 置信区间最大直线误差 (95th %ile)   : {p95_dist:.4f} 米")
    print("======================================================================")

    # 保存预测结果
    if save_pred_npy:
        pack = {
            "pred_raw": preds_raw_all,
            "gt_raw": gts_raw_all,
            "radar_xy": np.array(radar_xy, dtype=np.float32),
            "r_max": float(r_max),
        }
        np.save(pred_save_path, pack, allow_pickle=True)
        print(f"\n💾 测试结果已保存至: {pred_save_path}")

    return mean_dist_overall


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_size = (3, 224, 224)

    eval_best_model(
        device=device,
        image_size=image_size,
        weight_path="best_feat_model.pth",
        print_samples=3,  # 只打印3条供肉眼检查
        save_pred_npy=True,
        pred_save_path="test_predictions.npy",
        radar_xy=(200.0, 200.0),
        r_max=400.0
    )