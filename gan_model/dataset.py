import os
from mimetypes import guess_type
from multiprocessing import cpu_count

import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset

import utils


class ImageDataset(VisionDataset):
    def get_image_files(self) -> list:
        image_files = []
        for root, dirs, files in os.walk(self.root):
            for f in files:
                image_file = os.path.join(root, f)
                image_mime = guess_type(image_file)
                if image_mime is not None and image_mime[0][:6] == 'image/':
                    image_files.append(image_file)

        return image_files

    def __init__(self, root: str, transform=None, transparent: bool = False):
        super().__init__(root=root, transform=transform)
        self.image_files = self.get_image_files()
        self.transparent = transparent

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Image.Image:
        with open(self.image_files[idx], 'rb') as f:
            image = Image.open(f)
            if self.transparent and image.mode != 'RGBA':
                image = image.convert('RGBA')
            elif not self.transparent and image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.transform(image)

        return image


class SwapWH():
    def __init__(self):
        pass

    def __call__(self, x) -> torch.Tensor:
        x = torch.swapaxes(x, 1, 2)
        return x


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.image_path = args.image_path
        self.im_size = args.im_size
        self.transparent = args.transparent
        self.batch_size = args.batch_size

        self.width, self.height = self.im_size
        self.n_workers = cpu_count()
        self.transform = [
            transforms.ToTensor(),  # PIL to tensor and 0,255 to 0.0,1.0
        ]

        if utils.check_aspect_ratio(*self.im_size) != '1:1':
            self.transform.append(SwapWH())

        self.transform.extend([
            transforms.Resize((self.height, self.width)),
            transforms.Normalize(mean=(0.5), std=(0.5)),  # 0.0,1.0 to -1.0,1.0
        ])

        self.transform = transforms.Compose(self.transform)

        self.image_dataset = ImageDataset(
            self.image_path, self.transform, self.transparent
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.image_dataset, batch_size=self.batch_size, num_workers=self.n_workers,
            shuffle=True, drop_last=False, pin_memory=True
        )
