from pathlib import Path
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ========== 参数设置 ==========
DEFAULT_IMAGE_FOLDER_PATH = Path("JAM_data/")
DEFAULT_RESIZED_IMAGE_SIZE = 224


# ========== 图像预处理 ==========
def get_transforms(grayscale: bool = False, resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE):
    ops = [transforms.Resize((resize_size, resize_size))]
    if grayscale:
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.append(transforms.ToTensor())
    t = transforms.Compose(ops)
    return t, t


# ========== 双图像回归数据集（带归一化） ==========
class PairedImageRegressionDataset(Dataset):
    """
    CSV 示例：
    re_image,image,x1,y1,x2,y2
    JAM_data/test/re_images/xxx.png,JAM_data/test/images/xxx.png, 210,190, 350,220
    """
    def __init__(
        self,
        csv_file: Path,
        transform=None,
        radar_xy=(200.0, 200.0),
        r_max=400.0,
        normalize_target=True,
        path_is_relative_to_csv=True,
    ):
        self.transform = transform
        self.samples = []

        self.radar = np.array(radar_xy, dtype=np.float32)  # (2,)
        self.r_max = float(r_max)
        self.normalize_target = bool(normalize_target)
        self.csv_dir = csv_file.parent

        with csv_file.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"Header from CSV: {header}")

            for row in reader:
                if not row or all(not x.strip() for x in row):
                    continue

                img1_path = Path(row[0].strip())
                img2_path = Path(row[1].strip())

                # 解析相对路径：相对于 CSV 所在目录
                # ✅ 修正
                if path_is_relative_to_csv and not img1_path.is_absolute():
                    # 如果 CSV 里已经包含 JAM_data，就不要再拼
                    if img1_path.parts[0] != self.csv_dir.name:
                        img1_path = (self.csv_dir / img1_path).resolve()

                target = np.array([float(v) for v in row[2:]], dtype=np.float32)
                self.samples.append((img1_path, img2_path, target))

    def __len__(self):
        return len(self.samples)

    def _normalize_xy_pairs(self, target: np.ndarray) -> np.ndarray:
        """
        把 target 当作 (x,y) 对的序列归一化：
        (p - radar) / r_max
        例如 [x1,y1,x2,y2] -> 归一化后同维度
        """
        if target.ndim != 1 or target.size % 2 != 0:
            # 不是偶数维，就不做（避免错误）
            return target
        pts = target.reshape(-1, 2)  # [[x,y], ...]
        pts = (pts - self.radar[None, :]) / self.r_max
        return pts.reshape(-1)

    def __getitem__(self, idx):
        img1_path, img2_path, target = self.samples[idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.normalize_target:
            target = self._normalize_xy_pairs(target)

        return img1, img2, torch.tensor(target, dtype=torch.float32)


# ========== 数据包装类 ==========
class RegressionTaskData:
    def __init__(
        self,
        grayscale: bool = False,
        image_folder_path: Path = DEFAULT_IMAGE_FOLDER_PATH,
        resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE,
        train_csv: str = "re_train_paired.csv",
        test_csv: str = "re_test_paired.csv",
        radar_xy=(200.0, 200.0),
        r_max=400.0,
        normalize_target=True,
        batch_size=16,
        num_workers=0,
    ) -> None:
        self.grayscale = grayscale
        self.image_folder_path = Path(image_folder_path)
        self.resize_size = int(resize_size)

        self.radar_xy = radar_xy
        self.r_max = float(r_max)
        self.normalize_target = bool(normalize_target)

        self.train_transforms, self.test_transforms = get_transforms(grayscale, resize_size)

        self.trainset = self.make_dataset(train_csv, self.train_transforms)
        self.testset = self.make_dataset(test_csv, self.test_transforms)

        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    @property
    def output_image_size(self):
        return (1 if self.grayscale else 3, self.resize_size, self.resize_size)

    def make_dataset(self, csv_filename: str, transform) -> PairedImageRegressionDataset:
        return PairedImageRegressionDataset(
            csv_file=self.image_folder_path / csv_filename,
            transform=transform,
            radar_xy=self.radar_xy,
            r_max=self.r_max,
            normalize_target=self.normalize_target,
            path_is_relative_to_csv=True,
        )

    def denormalize_target(self, target: np.ndarray) -> np.ndarray:
        """
        把归一化后的 target 还原回真实坐标（按 (x,y) 对还原）
        """
        t = np.asarray(target, dtype=np.float32)
        if t.ndim != 1 or t.size % 2 != 0:
            return t
        pts = t.reshape(-1, 2) * self.r_max + np.array(self.radar_xy, dtype=np.float32)[None, :]
        return pts.reshape(-1)

    def visualise_image(self, index: int = 0, show_denorm: bool = True):
        inputs1, inputs2, targets = next(iter(self.testloader))

        img1 = inputs1[index].permute(1, 2, 0).numpy()
        img2 = inputs2[index].permute(1, 2, 0).numpy()
        target = targets[index].numpy()

        title = f"Target(norm): {np.round(target, 3)}"
        if show_denorm:
            title += f"\nTarget(denorm): {np.round(self.denormalize_target(target), 1)}"

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img1)
        axes[0].set_title("Input 1")
        axes[0].axis("off")

        axes[1].imshow(img2)
        axes[1].set_title("Input 2")
        axes[1].axis("off")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def data_image(self, index: int = 0):
        inputs1, inputs2, targets = next(iter(self.testloader))
        return inputs1[index], inputs2[index], targets[index]


if __name__ == "__main__":
    data = RegressionTaskData(
        train_csv="re_train_paired.csv",
        test_csv="re_test_paired.csv",
        radar_xy=(200.0, 200.0),
        r_max=400.0,
        normalize_target=True,
    )
    data.visualise_image()
