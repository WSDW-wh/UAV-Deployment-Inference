# ============================================================
# viz_test_idx_input_gt_pred_maps.py
#
# 功能：
# 1) 读取 test 集第 idx 个样本（inputs1）
# 2) 加载 best_feat_model.pth，输出预测位置（反归一化到米）
# 3) 用 GT 位置/Pred 位置各自生成一张“物理模型态势图”
# 4) 并排显示并保存： [inputs1 原始图 | GT 生成图 | Pred 生成图]
#
# 你只需要保证：
# - data_load_double.RegressionTaskData 能构建 testloader
# - utils.py 里有 jamm(), echo()
# - best_feat_model.pth 在同目录
# ============================================================

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from data_load_double import RegressionTaskData
from utils import jamm, echo
from PIL import Image
import time


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
# =========================
# Efficient Additive Attention
# =========================
class EfficientAdditiveAttention(nn.Module):
    def __init__(self, dim, token_dim=64, num_heads=2):
        super().__init__()
        self.to_query = nn.Linear(dim, token_dim * num_heads)
        self.to_key   = nn.Linear(dim, token_dim * num_heads)
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

        A = (q @ self.w_g) * self.scale
        A = F.normalize(A, dim=1)

        G = torch.sum(A * q, dim=1)
        G = einops.repeat(G, "b d -> b n d", n=k.shape[1])

        out = self.proj(G * k) + q
        out = self.final(out)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return out + residual


# =========================
# DiffConv
# =========================
class DiffConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.v  = nn.Conv2d(in_ch, out_ch, k, padding=p)
        self.c  = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)
        self.h  = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)
        self.vd = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)
        self.a  = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)

    def forward(self, x):
        return self.v(x) + self.c(x) + self.h(x) + self.vd(x) + self.a(x)


# =========================
# 主网络
# =========================
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

    def forward(self, x):
        x = self.pool1(self.attn1(F.relu(self.bn1(self.conv1(x)))))
        x = self.attn2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(x)

        y = x.flatten(1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


# =========================
# 反归一化：p = norm*r_max + radar
# =========================
def denorm_xy_pairs_torch(y_norm: torch.Tensor, radar_xy=(200.0, 200.0), r_max=400.0) -> torch.Tensor:
    if y_norm.ndim != 2 or (y_norm.shape[1] % 2 != 0):
        return y_norm
    radar = torch.tensor(radar_xy, dtype=y_norm.dtype, device=y_norm.device).view(1, 1, 2)
    pts = y_norm.view(y_norm.shape[0], -1, 2)  # [B, K, 2]
    pts = pts * float(r_max) + radar
    return pts.view(y_norm.shape[0], -1)


# =========================
# 生成态势图（返回 ndarray，不强制保存）
# =========================
def Jam2img_array(
    UAV1, UAV2,
    threshold=0.02,
    image_size=400,
    radar=(200.0, 200.0),
    radius=200.0,
    deep_color=(0, 0, 255)  # RGB
):
    H = W = image_size
    Radar = np.array(radar, dtype=float)
    UAV1  = np.asarray(UAV1, dtype=float)
    UAV2  = np.asarray(UAV2, dtype=float)

    # 图像坐标(y向下) -> 数学坐标(y向上)
    def img_to_math(x_img, y_img):
        return np.array([x_img, (H - 1) - y_img], dtype=float)

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    circle_mask = ((X - radar[0])**2 + (Y - radar[1])**2) <= radius**2
    ys, xs = np.where(circle_mask)

    out = np.ones((H, W, 3), dtype=np.float32) * 255.0
    deep = np.array(deep_color, dtype=np.float32)

    for y, x in zip(ys, xs):
        Target = img_to_math(x, y)
        Pj = jamm(Radar, Target, UAV1) + jamm(Radar, Target, UAV2)
        Pr = echo(Target, Radar)
        if Pr <= 0:
            continue
        jsr = Pj / Pr
        if jsr < threshold:
            severity = 1.0 - jsr / threshold
            severity = np.clip(severity, 0.0, 1.0)
            out[y, x, :] = (1 - severity) * 255.0 + severity * deep

    return out.astype(np.uint8)


# =========================
# 将 tensor 的 inputs1 转成可视化图片
# - 兼容两种常见情况：
#   (a) 已经是 0~1
#   (b) 是标准化后的（可能有负数/大于1），这里做 min-max 拉伸到 0~1
# =========================
def tensor_to_vis_rgb(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: [3,H,W] or [1,H,W]
    return: [H,W,3] uint8
    """
    x = img_t.detach().cpu().float()
    if x.ndim != 3:
        raise ValueError(f"Expect 3D tensor [C,H,W], got {x.shape}")

    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    # 先转到 [H,W,3]
    x = x.permute(1, 2, 0).contiguous().numpy()

    # 若数值不在[0,1]，做 min-max 归一化
    xmin, xmax = float(x.min()), float(x.max())
    if xmin < 0.0 or xmax > 1.0:
        if abs(xmax - xmin) < 1e-12:
            x = np.zeros_like(x)
        else:
            x = (x - xmin) / (xmax - xmin)

    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return x


# =========================
# 主接口：按 idx 读取 test 样本并出图
# =========================
def visualize_one_test_idx(
    idx=0,
    weight_path="best_feat_model.pth",
    save_dir="viz_out",
    image_size_net=(3, 224, 224),
    radar_xy=(200.0, 200.0),
    r_max=400.0,
    map_threshold=0.02,
    map_image_size=400,
    map_radius=200.0
):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据：与你训练一致
    data = RegressionTaskData(
        grayscale=(image_size_net[0] == 1),
        resize_size=image_size_net[1],
        radar_xy=radar_xy,
        r_max=r_max,
        normalize_target=True
    )

    # 取第 idx 个 test batch
    # 直接按“样本编号 idx”取 testset
    if idx < 0 or idx >= len(data.testset):
        raise IndexError(f"idx={idx} 超出 test 集范围: 0~{len(data.testset) - 1}")

    img1, img2, target = data.testset[idx]  # 每个都是单样本
    inputs1 = img1.unsqueeze(0).to(device)  # [1,3,224,224]
    targets = target.unsqueeze(0).to(device)  # [1,4]
    # 模型
    model = CNNRegression(image_size_net).to(device)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"❌ 找不到权重文件: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()


    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs1)
    start_time = time.perf_counter()
    with torch.no_grad():
        pred_norm = model(inputs1)          # [1,4]
        gt_norm = targets                   # [1,4]
        pred_raw = denorm_xy_pairs_torch(pred_norm, radar_xy=radar_xy, r_max=r_max)  # [1,4]
        gt_raw = denorm_xy_pairs_torch(gt_norm, radar_xy=radar_xy, r_max=r_max)      # [1,4]

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"算法运行时间: {execution_time:.6f} 秒")
    gt_raw_np = np.asarray(gt_raw.detach().cpu().numpy()).reshape(-1)
    pr_raw_np = np.asarray(pred_raw.detach().cpu().numpy()).reshape(-1)

    # 只取前 4 个（防止数据里多了维度）
    gt_raw_np = gt_raw_np[:4]
    pr_raw_np = pr_raw_np[:4]

    gt_uav1, gt_uav2 = gt_raw_np[0:2], gt_raw_np[2:4]
    pr_uav1, pr_uav2 = pr_raw_np[0:2], pr_raw_np[2:4]

    # 生成 GT / Pred 的“物理模型态势图”
    img_gt = Jam2img_array(
        gt_uav1, gt_uav2,
        threshold=map_threshold,
        image_size=map_image_size,
        radar=radar_xy,
        radius=map_radius
    )
    img_pr = Jam2img_array(
        pr_uav1, pr_uav2,
        threshold=map_threshold,
        image_size=map_image_size,
        radar=radar_xy,
        radius=map_radius
    )

    # inputs1 原始输入图（网络输入的那张）
    input_vis = tensor_to_vis_rgb(inputs1[0])   # 取 batch 内第 0 张

    pred_resized = np.array(
        Image.fromarray(img_pr).resize((input_vis.shape[1], input_vis.shape[0]), resample=Image.BILINEAR)
    )

    alpha = 0.45  # 透明度：0~1，越大 pred_map 越明显
    overlay = ( (1 - alpha) * input_vis.astype(np.float32) + alpha * pred_resized.astype(np.float32) )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    p_overlay = os.path.join(save_dir, f"idx{idx:04d}_input_pred_overlay.png")
    plt.imsave(p_overlay, overlay)

    # 保存单张图（可选）
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.6), dpi=300)
    axes = axes.ravel()

    images = [input_vis, img_gt, img_pr, overlay]
    captions = [
        "(a) Desired map",
        "(b) Ground truth",
        "(c) Prediction",
        "(d) Overlay"
    ]

    for ax, img, cap in zip(axes, images, captions):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

        # 子图说明放在下面
        ax.text(
            0.5, -0.035, cap,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=13
        )

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.985,
        bottom=0.06,
        wspace=0.06,
        hspace=0.16
    )

    p_quad = os.path.join(save_dir, f"idx{idx:04d}_quad.png")
    plt.savefig(p_quad, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.show()

    print("Saved:", p_overlay, p_quad)

    # 打印坐标，方便你核对
    print("\n===== Result =====")
    print("GT(norm):  ", gt_norm.squeeze(0).detach().cpu().numpy())
    print("PR(norm):  ", pred_norm.squeeze(0).detach().cpu().numpy())
    print("GT(raw m): ", gt_raw_np)
    print("PR(raw m): ", pr_raw_np)
    print("Saved:")
    print("==================")



    return {
        "idx": idx,
        "inputs1_vis": input_vis,
        "img_gt": img_gt,
        "img_pred": img_pr,
        "gt_raw": gt_raw_np,
        "pred_raw": pr_raw_np
    }


if __name__ == "__main__":
    # 改这里就能选择 test 的第几个样本
    visualize_one_test_idx(
        idx=10,
        weight_path="best_feat_model.pth",
        save_dir="viz_out",
        image_size_net=(3, 224, 224),
        radar_xy=(200.0, 200.0),
        r_max=400.0,
        map_threshold=0.02,
        map_image_size=400,
        map_radius=200.0
    )
