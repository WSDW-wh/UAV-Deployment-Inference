from typing import Any
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# ========== 参数设置 ==========
DEFAULT_IMAGE_FOLDER_PATH = Path('JAM_data/')
DEFAULT_CENTRE_CROP_SIZE = 224
DEFAULT_RESIZED_IMAGE_SIZE = 224


# ========== 图像预处理 ==========
def get_transforms(grayscale: bool = False, crop_size: int = DEFAULT_CENTRE_CROP_SIZE, resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE):
    if grayscale:
        t = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        return t, t
    else:
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        return t, t


# ========== 自定义双图像数据集 ==========
class PairedImageRegressionDataset(Dataset):
    def __init__(self, csv_file: Path, transform=None):
        self.transform = transform
        self.samples = []

        with csv_file.open('r') as f:
            lines = f.readlines()

        header = lines[0].strip().split(',')
        print(f"Header from CSV: {header}")

        for line in lines[1:]:
            parts = line.strip().split(',')
            img1_path = Path(parts[0])
            img2_path = Path(parts[1])
            target = np.array([float(x) for x in parts[2:]], dtype=np.float32)
            self.samples.append((img1_path, img2_path, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path, target = self.samples[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(target)


# ========== 数据包装类 ==========
class RegressionTaskData:
    def __init__(self,
                 grayscale: bool = False,
                 image_folder_path: Path = DEFAULT_IMAGE_FOLDER_PATH,
                 crop_size: int = DEFAULT_CENTRE_CROP_SIZE,
                 resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE,
                 train_csv: str = 'train.csv',
                 test_csv: str = 'test.csv'
                 ) -> None:
        self.grayscale = grayscale
        self.image_folder_path = image_folder_path
        self.train_transforms, self.test_transforms = get_transforms(grayscale, crop_size, resize_size)
        self.trainloader = self.make_loader(train_csv, self.train_transforms)
        self.testloader = self.make_loader(test_csv, self.test_transforms)
        self.crop_size = crop_size
        self.resize_size = resize_size

    @property
    def output_image_size(self):
        return (1 if self.grayscale else 3, self.resize_size, self.resize_size)

    def make_loader(self, csv_filename, transform) -> DataLoader:
        dataset = PairedImageRegressionDataset(
            csv_file=self.image_folder_path / csv_filename,
            transform=transform
        )
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def visualise_image(self, index: int = 0):
        """
        可视化某一个 batch 中的第 index 张图像对（inputs1 和 inputs2），并显示目标值。
        """
        inputs1, inputs2, targets = next(iter(self.testloader))

        img1 = inputs1[index]
        img2 = inputs2[index]
        target = targets[index].numpy()

        # 转换为图像可视化格式
        img1 = img1.permute(1, 2, 0).numpy()
        img2 = img2.permute(1, 2, 0).numpy()

        # 创建并排显示
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img1)
        axes[0].set_title("Input 1")
        axes[0].axis('off')

        axes[1].imshow(img2)
        axes[1].set_title("Input 2")
        axes[1].axis('off')

        # 显示目标值
        fig.suptitle(f"Target: {np.round(target, 2)}", fontsize=12)
        plt.tight_layout()
        plt.show()

    def data_image(self, index):
        inputs1, inputs2, targets = next(iter(self.testloader))
        return inputs1[index], inputs2[index], targets[index]


# ========== 示例运行 ==========
if __name__ == '__main__':
    data = RegressionTaskData()
    data.visualise_image()
