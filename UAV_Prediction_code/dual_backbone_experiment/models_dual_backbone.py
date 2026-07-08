import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
        b, c, h, w = x.shape

        tokens = x.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]

        q = F.normalize(self.to_query(tokens), dim=-1)
        k = F.normalize(self.to_key(tokens), dim=-1)

        a = (q @ self.w_g) * self.scale
        a = F.normalize(a, dim=1)

        g = torch.sum(a * q, dim=1)
        g = einops.repeat(g, "b d -> b n d", n=k.shape[1])

        out = self.proj(g * k) + q
        out = self.final(out)

        out = out.permute(0, 2, 1).contiguous().view(b, c, h, w)
        return out + residual


class PlainConvBackbone(nn.Module):
    def __init__(self, in_ch=3, use_attention=True, embed_dim=256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.attn = EfficientAdditiveAttention(256) if use_attention else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Identity() if embed_dim == 256 else nn.Linear(256, embed_dim)
        self.out_dim = embed_dim

    def forward(self, x, return_map=False):
        feat_map = self.stem(x)
        feat_map = self.attn(feat_map)
        feat_vec = self.pool(feat_map).flatten(1)
        feat_vec = self.proj(feat_vec)

        if return_map:
            return feat_vec, feat_map
        return feat_vec


class ResNet18FeatureBackbone(nn.Module):
    def __init__(self, use_attention=True, pretrained=False, embed_dim=256):
        super().__init__()

        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
        except Exception:
            net = models.resnet18(pretrained=pretrained)

        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )

        self.attn = EfficientAdditiveAttention(512) if use_attention else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Identity() if embed_dim == 512 else nn.Linear(512, embed_dim)
        self.out_dim = embed_dim

    def forward(self, x, return_map=False):
        feat_map = self.stem(x)
        feat_map = self.attn(feat_map)
        feat_vec = self.pool(feat_map).flatten(1)
        feat_vec = self.proj(feat_vec)

        if return_map:
            return feat_vec, feat_map
        return feat_vec


class RegressorHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SingleBackboneRegressor(nn.Module):
    """
    单输入版本：
    输入 x -> backbone -> regression head
    """
    def __init__(
        self,
        backbone_type="plaincnn",
        use_attention=True,
        pretrained=False,
        out_dim=4,
        embed_dim=256,
    ):
        super().__init__()

        if backbone_type == "resnet18":
            self.backbone = ResNet18FeatureBackbone(
                use_attention=use_attention,
                pretrained=pretrained,
                embed_dim=embed_dim,
            )
        elif backbone_type == "plaincnn":
            self.backbone = PlainConvBackbone(
                in_ch=3,
                use_attention=use_attention,
                embed_dim=embed_dim,
            )
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        self.head = RegressorHead(
            in_dim=self.backbone.out_dim,
            hidden_dim=256,
            out_dim=out_dim,
        )

    def forward(self, x, return_features=False):
        feat = self.backbone(x)
        pred = self.head(feat)
        if return_features:
            return pred, feat
        return pred


class DualBackboneRegressor(nn.Module):
    """
    dual-input 里的 student 分支
    """
    def __init__(
        self,
        backbone_type="plaincnn",
        use_attention=True,
        pretrained=False,
        out_dim=4,
        embed_dim=256,
    ):
        super().__init__()

        if backbone_type == "resnet18":
            self.backbone = ResNet18FeatureBackbone(
                use_attention=use_attention,
                pretrained=pretrained,
                embed_dim=embed_dim,
            )
        elif backbone_type == "plaincnn":
            self.backbone = PlainConvBackbone(
                in_ch=3,
                use_attention=use_attention,
                embed_dim=embed_dim,
            )
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        self.head = RegressorHead(
            in_dim=self.backbone.out_dim,
            hidden_dim=256,
            out_dim=out_dim,
        )

    def forward(self, x, return_features=False):
        feat = self.backbone(x)
        pred = self.head(feat)
        if return_features:
            return pred, feat
        return pred


class DualFeatureTeacher(nn.Module):
    """
    dual-input 里的 teacher 分支，只输出特征
    """
    def __init__(
        self,
        backbone_type="plaincnn",
        use_attention=True,
        pretrained=False,
        embed_dim=256,
    ):
        super().__init__()

        if backbone_type == "resnet18":
            self.backbone = ResNet18FeatureBackbone(
                use_attention=use_attention,
                pretrained=pretrained,
                embed_dim=embed_dim,
            )
        elif backbone_type == "plaincnn":
            self.backbone = PlainConvBackbone(
                in_ch=3,
                use_attention=use_attention,
                embed_dim=embed_dim,
            )
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

    def forward(self, x):
        return self.backbone(x)