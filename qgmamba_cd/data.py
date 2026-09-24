from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import DataConfig
from .distributed import DistributedContext, seed_worker


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SplitPaths:
    a: list[Path]
    b: list[Path]
    labels: list[Path]
    names: list[str]

    def __len__(self) -> int:
        return len(self.names)


class PairedTrainTransform:
    """Shared geometry with a deliberately independent temporal appearance noise."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.geometry = A.Compose(
            [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),
            ],
            additional_targets={"image2": "image"},
        )
        self.normalize = A.Compose(
            [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2(transpose_mask=True)],
            additional_targets={"image2": "image"},
        )

    @staticmethod
    def _appearance(image: np.ndarray, cfg: DataConfig) -> np.ndarray:
        output = image.astype(np.float32)
        if random.random() < cfg.independent_photometric_prob:
            contrast = random.uniform(0.8, 1.2)
            brightness = random.uniform(-25.0, 25.0)
            output = output * contrast + brightness
            hsv = cv2.cvtColor(np.clip(output, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.8, 1.2), 0, 255)
            output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            if random.random() < 0.25:
                output += np.random.normal(0.0, random.uniform(2.0, 8.0), output.shape)
            if random.random() < 0.15:
                output = cv2.GaussianBlur(output, (3, 3), 0)
        if random.random() < cfg.shadow_prob:
            height, width = output.shape[:2]
            shadow = np.ones((height, width), dtype=np.float32)
            points = np.asarray(
                [
                    [random.randint(0, width - 1), random.randint(0, height - 1)]
                    for _ in range(random.randint(3, 6))
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(shadow, [points], random.uniform(0.55, 0.85))
            shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=5.0)
            output *= shadow[..., None]
        output = np.clip(output, 0, 255).astype(np.uint8)
        if random.random() < cfg.compression_prob:
            quality = random.randint(45, 90)
            ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if ok:
                output = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return output

    def __call__(self, *, image: np.ndarray, image2: np.ndarray, mask: np.ndarray):
        transformed = self.geometry(image=image, image2=image2, mask=mask)
        image_a = self._appearance(transformed["image"], self.cfg)
        image_b = self._appearance(transformed["image2"], self.cfg)
        return self.normalize(image=image_a, image2=image_b, mask=transformed["mask"])


def build_transforms(cfg: DataConfig) -> tuple[PairedTrainTransform, A.Compose]:
    train = PairedTrainTransform(cfg)
    evaluate = A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(transpose_mask=True),
        ],
        additional_targets={"image2": "image"},
    )
    return train, evaluate


def _resolve_name(directory: Path, name: str) -> tuple[Path, str]:
    supplied = Path(name).name
    if Path(supplied).suffix:
        return directory / supplied, supplied
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = directory / f"{supplied}{suffix}"
        if candidate.exists():
            return candidate, candidate.name
    return directory / f"{supplied}.png", f"{supplied}.png"


def parse_split(root: str | Path, split: str) -> SplitPaths:
    root = Path(root).expanduser().resolve()
    list_path = root / "list" / f"{split}.txt"
    required_dirs = [root / "A", root / "B", root / "label"]
    missing_layout = [str(p) for p in [list_path, *required_dirs] if not p.exists()]
    if missing_layout:
        raise FileNotFoundError(
            "LEVIR-CD256 layout is incomplete. Missing: " + ", ".join(missing_layout)
        )
    names_raw = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    a_paths: list[Path] = []
    b_paths: list[Path] = []
    label_paths: list[Path] = []
    names: list[str] = []
    for raw_name in names_raw:
        a_path, filename = _resolve_name(root / "A", raw_name)
        b_path, _ = _resolve_name(root / "B", filename)
        label_path, _ = _resolve_name(root / "label", filename)
        a_paths.append(a_path)
        b_paths.append(b_path)
        label_paths.append(label_path)
        names.append(filename)
    missing_files = [str(p) for paths in (a_paths, b_paths, label_paths) for p in paths if not p.exists()]
    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(f"Split '{split}' references missing files (first 10):\n{preview}")
    if not names:
        raise ValueError(f"Split list is empty: {list_path}")
    return SplitPaths(a_paths, b_paths, label_paths, names)


def subset_split(paths: SplitPaths, fraction: float) -> SplitPaths:
    if fraction >= 1.0:
        return paths
    count = max(1, int(len(paths) * fraction))
    return SplitPaths(paths.a[:count], paths.b[:count], paths.labels[:count], paths.names[:count])


@dataclass(frozen=True)
class Readers:
    read_image: Callable[[Path], np.ndarray]
    read_mask: Callable[[Path], np.ndarray]


def read_rgb_cv2(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_grayscale_mask_reader(threshold: int = 127) -> Callable[[Path], np.ndarray]:
    def _read(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"OpenCV could not read mask: {path}")
        return (mask > threshold).astype(np.uint8)
    return _read


DEFAULT_READERS = Readers(read_image=read_rgb_cv2, read_mask=make_grayscale_mask_reader(127))

IndexBuilder = Callable[["str | Path", str], "SplitPaths"]

# Private aliases: kept so any other internal call sites in this file keep working unchanged.
_read_rgb = read_rgb_cv2
_read_mask = DEFAULT_READERS.read_mask


class PairedPatchDataset(Dataset):
    def __init__(
        self,
        paths: SplitPaths,
        patch_size: int,
        transform,
        epoch_multiplier: int = 1,
        change_focus_prob: float = 0.6,
        min_change_ratio: float = 0.005,
        max_tries: int = 25,
        temporal_swap_prob: float = 0.0,
        *,
        readers: Readers = DEFAULT_READERS,
    ) -> None:
        self.paths = paths
        self.patch_size = patch_size
        self.transform = transform
        self.epoch_multiplier = epoch_multiplier
        self.change_focus_prob = change_focus_prob
        self.min_change_ratio = min_change_ratio
        self.max_tries = max_tries
        self.temporal_swap_prob = temporal_swap_prob
        self.readers = readers

    def __len__(self) -> int:
        return len(self.paths) * self.epoch_multiplier

    def _sample_xy(self, mask: np.ndarray) -> tuple[int, int]:
        height, width = mask.shape
        patch = self.patch_size
        if height < patch or width < patch:
            raise ValueError(
                f"Image {height}x{width} is smaller than configured patch size {patch}. "
                "Use a smaller data.image_size."
            )
        if height == patch and width == patch:
            return 0, 0
        if random.random() < self.change_focus_prob:
            for _ in range(self.max_tries):
                y = random.randint(0, height - patch)
                x = random.randint(0, width - patch)
                if mask[y : y + patch, x : x + patch].mean() >= self.min_change_ratio:
                    return y, x
        return random.randint(0, height - patch), random.randint(0, width - patch)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index %= len(self.paths)
        image_a = self.readers.read_image(self.paths.a[index])
        image_b = self.readers.read_image(self.paths.b[index])
        mask = self.readers.read_mask(self.paths.labels[index])
        y, x = self._sample_xy(mask)
        patch = self.patch_size
        image_a = image_a[y : y + patch, x : x + patch]
        image_b = image_b[y : y + patch, x : x + patch]
        mask = mask[y : y + patch, x : x + patch]
        if random.random() < self.temporal_swap_prob:
            image_a, image_b = image_b, image_a
        if self.transform is not None:
            transformed = self.transform(image=image_a, image2=image_b, mask=mask)
            tensor_mask = transformed["mask"]
            if tensor_mask.ndim == 2:
                tensor_mask = tensor_mask.unsqueeze(0)
            return transformed["image"].float(), transformed["image2"].float(), tensor_mask.float()
        return (
            torch.from_numpy(image_a.transpose(2, 0, 1)).float() / 255.0,
            torch.from_numpy(image_b.transpose(2, 0, 1)).float() / 255.0,
            torch.from_numpy(mask).unsqueeze(0).float(),
        )


LEVIRPatchDataset = PairedPatchDataset


class PairedFullDataset(Dataset):
    def __init__(
        self,
        paths: SplitPaths,
        transform: A.Compose | None,
        *,
        readers: Readers = DEFAULT_READERS,
    ) -> None:
        self.paths = paths
        self.transform = transform
        self.readers = readers

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, str]:
        image_a = self.readers.read_image(self.paths.a[index])
        image_b = self.readers.read_image(self.paths.b[index])
        mask = self.readers.read_mask(self.paths.labels[index])
        if self.transform is not None:
            transformed = self.transform(image=image_a, image2=image_b, mask=mask)
            return transformed["image"].float(), transformed["image2"].float(), mask, self.paths.names[index]
        return (
            torch.from_numpy(image_a.transpose(2, 0, 1)).float() / 255.0,
            torch.from_numpy(image_b.transpose(2, 0, 1)).float() / 255.0,
            mask,
            self.paths.names[index],
        )


LEVIRFullDataset = PairedFullDataset


@dataclass
class DatasetBundle:
    train_dataset: PairedPatchDataset
    train_loader: DataLoader
    train_sampler: DistributedSampler | None
    val_dataset: PairedFullDataset
    test_dataset: PairedFullDataset
    train_paths: SplitPaths
    readers: Readers


def build_datasets(
    cfg: DataConfig,
    context: DistributedContext,
    seed: int,
    batch_size: int,
    index_builder: IndexBuilder = parse_split,
    readers: Readers = DEFAULT_READERS,
) -> DatasetBundle:
    train_transform, eval_transform = build_transforms(cfg)
    train_paths = subset_split(index_builder(cfg.root, "train"), cfg.subset_fraction)
    val_paths = subset_split(index_builder(cfg.root, "val"), cfg.subset_fraction)
    test_paths = subset_split(index_builder(cfg.root, "test"), cfg.subset_fraction)
    train_dataset = PairedPatchDataset(
        train_paths,
        patch_size=cfg.image_size,
        transform=train_transform,
        epoch_multiplier=cfg.train_multiplier,
        change_focus_prob=cfg.change_focus_prob,
        min_change_ratio=cfg.min_change_ratio,
        max_tries=cfg.max_crop_tries,
        temporal_swap_prob=cfg.temporal_swap_prob,
        readers=readers,
    )
    sampler = None
    if context.distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
    generator = torch.Generator()
    generator.manual_seed(seed + context.rank)
    loader_options = dict(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if cfg.num_workers > 0:
        loader_options["prefetch_factor"] = cfg.prefetch_factor
    loader = DataLoader(**loader_options)
    if not loader:
        raise ValueError(
            "Training loader has zero batches. Reduce train.batch_size or increase data.train_multiplier."
        )
    return DatasetBundle(
        train_dataset,
        loader,
        sampler,
        PairedFullDataset(val_paths, eval_transform, readers=readers),
        PairedFullDataset(test_paths, eval_transform, readers=readers),
        train_paths,
        readers,
    )
