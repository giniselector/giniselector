import inspect
from typing import Any, Callable, Optional, Type, Union

import torch
import torch.utils.data
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import get_dataset_cls


class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, ds: torch.utils.data.Dataset) -> None:
        self.ds = ds

    def __getitem__(self, index):
        x, y = self.ds.__getitem__(index)
        return x, y, index

    def __len__(self):
        return len(self.ds)


class DefaultDataModule(VisionDataModule):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        num_workers: int = 0,
        val_split: Union[int, float] = 0.2,
        batch_size: int = 32,
        seed: int = 42,
        pin_memory: bool = True,
        train_shuffle: bool = True,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        get_data_index=False,
        *args,
        **kwargs,
    ):
        shuffle = kwargs.pop("shuffle", False)
        drop_last = kwargs.pop("drop_last", False)
        super().__init__(
            data_dir,
            val_split,
            num_workers,
            True,
            batch_size,
            seed,
            shuffle,
            pin_memory,
            drop_last,
            *args,
            **kwargs,
        )
        self.name = dataset_name
        self.train_shuffle = train_shuffle
        self.get_data_index = get_data_index

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.transforms = test_transforms

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.__post_init__()

    def __post_init__(self):
        self.dataset_cls = get_dataset_cls(self.name)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Prepares the data."""
        self.get_dataset_train()
        self.get_dataset_test()

    def get_dataset_train(self, transform=None, *args, **kwargs):
        if (
            "split" in inspect.getfullargspec(self.dataset_cls.__init__)[0]
            or "split" in inspect.getfullargspec(self.dataset_cls)[0]
        ):
            ds = self.dataset_cls(
                self.data_dir,
                split="train",
                download=True,
                transform=transform,
                *args,
                **kwargs,
            )
        else:
            ds = self.dataset_cls(
                self.data_dir,
                train=True,
                download=True,
                transform=transform,
                *args,
                **kwargs,
            )

        if self.get_data_index:
            ds = DatasetWithIndex(ds)

        return ds

    def get_dataset_test(self, transform=None, *args, **kwargs):
        if (
            "split" in inspect.getfullargspec(self.dataset_cls.__init__)[0]
            or "split" in inspect.getfullargspec(self.dataset_cls)[0]
        ):
            ds = self.dataset_cls(
                self.data_dir,
                split="test",
                download=True,
                transform=transform,
                **self.EXTRA_ARGS,
            )
        else:
            ds = self.dataset_cls(
                self.data_dir,
                train=False,
                download=True,
                transform=transform,
                **self.EXTRA_ARGS,
            )
        if self.get_data_index:
            ds = DatasetWithIndex(ds)

        return ds

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage in ["fit", "val"] or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.get_dataset_train(train_transforms)
            self.dataset_train = self._split_dataset(dataset_train)

            if self.val_split > 0:
                dataset_val = self.get_dataset_train(val_transforms)
                self.dataset_val = self._split_dataset(dataset_val, train=False)
            else:
                self.dataset_val = self.get_dataset_test(val_transforms)

        if stage in ["test", "predict"] or stage is None:
            test_transforms = self.default_transforms() if self.transforms is None else self.transforms
            self.dataset_test = self.get_dataset_test(test_transforms)

    def default_transforms(self) -> Callable:
        """Default transform for the dataset."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The prediction dataloader."""
        return self._dataloader(self.dataset_test)


class TrainValTestDataModule(DefaultDataModule):
    def get_dataset_val(self, transform=None):
        return self.dataset_cls(
            self.data_dir,
            split="val",
            download=True,
            transform=transform,
            **self.EXTRA_ARGS,
        )

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Prepares the data."""
        self.get_dataset_train()
        self.get_dataset_val()
        self.get_dataset_test()

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage in ["fit", "val"] or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            self.dataset_train = self.get_dataset_train(train_transforms)
            self.dataset_val = self.get_dataset_val(val_transforms)

        if stage in ["test", "predict"] or stage is None:
            test_transforms = self.default_transforms() if self.transforms is None else self.transforms
            self.dataset_test = self.get_dataset_test(test_transforms)


vision_datamodules_registry = {
    "default": DefaultDataModule,
    "train_val_test": TrainValTestDataModule,
}


def get_vision_datamodule_cls(name: str = "default") -> Type[VisionDataModule]:
    if name is not None:
        return vision_datamodules_registry[name]
    return vision_datamodules_registry["default"]


def get_vision_datamodules_names():
    return list(vision_datamodules_registry.keys())
