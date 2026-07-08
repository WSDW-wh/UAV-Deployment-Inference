import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import einops
from data_load_double import RegressionTaskData


# =====================================================
# Efficient Additive Attention
# =====================================================
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
        """
        x: [B, C, H, W]
        """
        residual = x                    # ✅ 保存原始 feature map
        B, C, H, W = x.shape

        x_tokens = x.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        q = F.normalize(self.to_query(x_tokens), dim=-1)
        k = F.normalize(self.to_key(x_tokens), dim=-1)

        A = (q @ self.w_g) * self.scale               # [B, N, 1]
        A = F.normalize(A, dim=1)

        G = torch.sum(A * q, dim=1)                   # [B, D]
        G = einops.repeat(G, 'b d -> b n d', n=k.shape[1])

        out = self.proj(G * k) + q
        out = self.final(out)                          # [B, N, C]

        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return out + residual                          # ✅ 正确 residual



# =====================================================
# DiffConv
# =====================================================
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


# =====================================================
# 主网络（inputs1 = re_images）
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
        features = self.pool2(x)              # 🔹 特征监督点

        y = features.flatten(1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        if return_features:
            return y, features
        return y


# =====================================================
# Teacher / 特征网络（inputs2 = images）
# =====================================================
class FeatureExtractor(nn.Module):
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

    def forward(self, x):
        x = self.pool1(self.attn1(F.relu(self.bn1(self.conv1(x)))))
        x = self.attn2(F.relu(self.bn2(self.conv2(x))))
        return self.pool2(x)


# =====================================================
# 训练 + 评估（测试只用 inputs1）
# =====================================================
def train_and_eval(device,
                   n_epochs=50,
                   image_size=(3, 224, 224),
                   lambda_feat=10.0,
                   test_every=5):

    regression_task = RegressionTaskData(
        grayscale=(image_size[0] == 1),
        resize_size=image_size[1]
    )

    model = CNNRegression(image_size).to(device)
    teacher = FeatureExtractor(image_size).to(device)

    # ✅ teacher 也参与训练
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(teacher.parameters()),
        lr=3e-4
    )

    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.5)
    writer = SummaryWriter()

    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(n_epochs):
        model.train()
        teacher.train()

        for i, (inputs1, inputs2, targets) in enumerate(regression_task.trainloader):
            inputs1 = inputs1.to(device)   # re_images
            inputs2 = inputs2.to(device)   # images
            targets = targets.to(device)

            optimizer.zero_grad()

            preds, feat1 = model(inputs1, return_features=True)
            feat2 = teacher(inputs2)

            loss_reg = criterion(preds, targets)
            loss_feat = 1.0 - F.cosine_similarity(
                feat1.flatten(1), feat2.flatten(1), dim=1
            ).mean()

            loss = loss_reg + lambda_feat * loss_feat
            loss.backward()
            optimizer.step()

            step = epoch * len(regression_task.trainloader) + i
            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/Loss_Reg", loss_reg.item(), step)
            writer.add_scalar("Train/Loss_Feat", loss_feat.item(), step)

        scheduler.step()

        # ===== 测试（只用 inputs1）=====
        if (epoch + 1) % test_every == 0:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for inputs1, _, targets in regression_task.testloader:
                    pred = model(inputs1.to(device))
                    total += criterion(pred, targets.to(device)).item()
            test_loss = total / len(regression_task.testloader)

            print(f"[Eval] Epoch {epoch+1}: TestLoss(MSE,norm)={test_loss:.6f}")
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), "best_feat_model.pth")
                print(f"✅ Best model saved at epoch {best_epoch}")

    writer.close()
    return model, best_epoch, best_loss


# =====================================================
# 主入口
# =====================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_size = (3, 224, 224)

    model, ep, loss = train_and_eval(
        device,
        n_epochs=120,
        image_size=image_size,
        lambda_feat=10.0
    )

    print(f"🎯 Training done. Best epoch={ep}, loss={loss:.6f}")
