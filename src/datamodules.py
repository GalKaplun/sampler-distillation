import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.cifar5m import CIFAR5m
from src.data_utils import CustomDataset


def get_cifar5m():

    mean = (0.4555, 0.4362, 0.3415)
    std = (0.2284, 0.2167, 0.2165)
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return CIFAR5m(data_dir=None, transform=transform)


def get_cifar(data_path, train, no_transform=False):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    if no_transform:
        transform = None
    return CIFAR10(data_path, train=train, transform=transform, download=True)


def get_noisy_cifar(data_path):
    reg_cifar = get_cifar(data_path, train=True, no_transform=True)

    targets = np.fromfile('./files/noisy_labels.npy',
                          dtype='int64')
    mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return CustomDataset(data_path, reg_cifar, targets=targets, original_targets=reg_cifar.targets,
                         transform=transform)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        batch_size: int = 256,
        num_workers: int = 4,
        add_noise: bool = False,
        dataset: str = 'cifar10'
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_noise = add_noise
        self.dataset = dataset

    def setup(self, stage=None):
        if self.dataset.lower() == 'cifar10':
            if self.add_noise:
                self.train_set = get_noisy_cifar(data_path=self.data_dir)
            else:
                self.train_set = get_cifar(data_path=self.data_dir, train=True)
        elif self.dataset.lower() == 'cifar5m':
            self.train_set = get_cifar5m()
        self.val_set = get_cifar(data_path=self.data_dir, train=False)

    def train_dataloader(self):
        print(self.batch_size)
        print(self.num_workers)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          prefetch_factor=2)
