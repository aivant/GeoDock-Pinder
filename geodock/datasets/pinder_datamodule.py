from typing import Optional, Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate
from geodock.datasets.pinder_dataset import PinderDataset

class PinderDataModule(pl.LightningDataModule):
    def __init__(self, data_train: PinderDataset, collate_fn: Optional[Callable]=None):
        super().__init__()
        self.data_train = data_train
        self.collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, collate_fn=self.collate_fn)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, collate_fn=self.collate_fn)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, collate_fn=self.collate_fn)
