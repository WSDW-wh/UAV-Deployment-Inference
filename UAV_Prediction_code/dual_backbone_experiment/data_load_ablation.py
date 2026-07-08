from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_IMAGE_FOLDER_PATH = PROJECT_ROOT / "JAM_data"
DEFAULT_RESIZED_IMAGE_SIZE = 224


def get_transforms(grayscale: bool = False, resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE):
    ops = [transforms.Resize((resize_size, resize_size))]
    if grayscale:
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.append(transforms.ToTensor())
    t = transforms.Compose(ops)
    return t, t


class _BaseRegressionDataset(Dataset):
    def __init__(
        self,
        transform=None,
        radar_xy=(200.0, 200.0),
        r_max=400.0,
        normalize_target=True,
    ):
        self.transform = transform
        self.samples = []
        self.radar = np.array(radar_xy, dtype=np.float32)
        self.r_max = float(r_max)
        self.normalize_target = bool(normalize_target)

    def __len__(self):
        return len(self.samples)

    def _normalize_xy_pairs(self, target: np.ndarray) -> np.ndarray:
        if target.ndim != 1 or target.size % 2 != 0:
            return target
        pts = target.reshape(-1, 2)
        pts = (pts - self.radar[None, :]) / self.r_max
        return pts.reshape(-1)

    def denormalize_target(self, target: np.ndarray) -> np.ndarray:
        t = np.asarray(target, dtype=np.float32)
        if t.ndim != 1 or t.size % 2 != 0:
            return t
        pts = t.reshape(-1, 2) * self.r_max + self.radar[None, :]
        return pts.reshape(-1)

    def _resolve_path(self, img_path: Path, csv_dir: Path) -> Path:
        if img_path.is_absolute():
            return img_path
        if len(img_path.parts) > 0 and img_path.parts[0] == "JAM_data":
            return (PROJECT_ROOT / img_path).resolve()
        if len(img_path.parts) > 0 and img_path.parts[0] in ("train", "test"):
            return (DEFAULT_IMAGE_FOLDER_PATH / img_path).resolve()
        return (csv_dir / img_path).resolve()


class PairedImageRegressionDataset(_BaseRegressionDataset):
    """
    CSV format:
    re_image,image,x,y,a,b
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
        super().__init__(transform, radar_xy, r_max, normalize_target)
        self.csv_dir = csv_file.parent

        with csv_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"[Paired] Header from CSV: {header}")
            for row in reader:
                if not row or all(not x.strip() for x in row):
                    continue
                img1_path = Path(row[0].strip())
                img2_path = Path(row[1].strip())
                if path_is_relative_to_csv:
                    img1_path = self._resolve_path(img1_path, self.csv_dir)
                    img2_path = self._resolve_path(img2_path, self.csv_dir)
                target = np.array([float(v) for v in row[2:]], dtype=np.float32)
                self.samples.append((img1_path, img2_path, target))

    def __getitem__(self, idx):
        img1_path, img2_path, target = self.samples[idx]
        if not img1_path.exists():
            raise FileNotFoundError(f"Image not found: {img1_path}")
        if not img2_path.exists():
            raise FileNotFoundError(f"Image not found: {img2_path}")

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.normalize_target:
            target = self._normalize_xy_pairs(target)
        return img1, img2, torch.tensor(target, dtype=torch.float32)


class RegressionTaskDataDual:
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
        self.trainloader = self.make_loader(train_csv, self.train_transforms, batch_size, True, num_workers)
        self.testloader = self.make_loader(test_csv, self.test_transforms, batch_size, False, num_workers)

    @property
    def output_image_size(self):
        return (1 if self.grayscale else 3, self.resize_size, self.resize_size)

    def make_loader(self, csv_filename: str, transform, batch_size: int, shuffle: bool, num_workers: int):
        dataset = PairedImageRegressionDataset(
            csv_file=self.image_folder_path / csv_filename,
            transform=transform,
            radar_xy=self.radar_xy,
            r_max=self.r_max,
            normalize_target=self.normalize_target,
            path_is_relative_to_csv=True,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
class SingleImageWrapper(torch.utils.data.Dataset):
    """
    从 Paired 数据集中提取单输入
    use_column:
        0 -> re_image
        1 -> image
    """
    def __init__(self, dataset, use_column=0):
        self.dataset = dataset
        self.use_column = use_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, img2, target = self.dataset[idx]

        if self.use_column == 0:
            return img1, target
        else:
            return img2, target


class RegressionTaskDataSingle:
    """
    单输入版本（基于 Paired 数据封装）
    """

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
        use_column=0,
    ) -> None:

        self.grayscale = grayscale
        self.image_folder_path = Path(image_folder_path)
        self.resize_size = int(resize_size)
        self.radar_xy = radar_xy
        self.r_max = float(r_max)
        self.normalize_target = bool(normalize_target)

        self.train_transforms, self.test_transforms = get_transforms(grayscale, resize_size)

        self.trainloader = self.make_loader(
            train_csv, self.train_transforms, batch_size, True, num_workers, use_column
        )
        self.testloader = self.make_loader(
            test_csv, self.test_transforms, batch_size, False, num_workers, use_column
        )

    def make_loader(
        self,
        csv_filename,
        transform,
        batch_size,
        shuffle,
        num_workers,
        use_column,
    ):
        paired_dataset = PairedImageRegressionDataset(
            csv_file=self.image_folder_path / csv_filename,
            transform=transform,
            radar_xy=self.radar_xy,
            r_max=self.r_max,
            normalize_target=self.normalize_target,
            path_is_relative_to_csv=True,
        )

        dataset = SingleImageWrapper(paired_dataset, use_column)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )