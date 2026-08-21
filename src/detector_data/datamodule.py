from __future__ import annotations

from collections import OrderedDict

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .factory import build_single_dataset
from .mixed import WeightedMixedDataset
from .transforms import build_transform


class RIFTDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None

    def setup(self, stage: str | None = None):
        if stage in (None, "fit") and self.train_set is None and hasattr(self.cfg, "dataset"):
            t = build_transform(self.cfg.transform, train=True)
            if str(self.cfg.dataset.name) == "mixed":
                datasets = OrderedDict()
                weights = OrderedDict()
                for d in self.cfg.dataset.mixed.domains:
                    name = str(d.name)
                    key = str(d.config_key)
                    split = str(getattr(d, "split", "train"))
                    domain_cfg = self.cfg.dataset[key]
                    datasets[name] = build_single_dataset(name, domain_cfg, t, split=split)
                    weights[name] = float(d.weight)
                self.train_set = WeightedMixedDataset(
                    datasets,
                    weights,
                    total_per_epoch=int(self.cfg.dataset.mixed.total_per_epoch),
                    seed=int(self.cfg.seed),
                )
            else:
                name = str(self.cfg.dataset.name)
                domain_cfg = self.cfg.dataset[name]
                split = str(getattr(domain_cfg, "split", "train"))
                self.train_set = build_single_dataset(name, domain_cfg, t, split=split)

        if stage in (None, "fit", "validate") and self.val_set is None:
            tval = build_transform(self.cfg.transform, train=False)
            vcfg = self.cfg.val_dataset
            name = str(vcfg.name)
            domain_cfg = vcfg[name]
            split = str(getattr(domain_cfg, "split", "val"))
            self.val_set = build_single_dataset(name, domain_cfg, tval, split=split)

    def set_epoch(self, epoch: int):
        if hasattr(self.train_set, "set_epoch"):
            self.train_set.set_epoch(epoch)

    def _loader(self, ds, *, train: bool):
        c = self.cfg.loader
        kwargs = dict(
            dataset=ds,
            batch_size=int(c.batch_size),
            num_workers=int(c.num_workers),
            pin_memory=bool(c.pin_memory),
            drop_last=bool(c.drop_last) if train else False,
            shuffle=train,
        )
        if int(c.num_workers) > 0:
            kwargs["persistent_workers"] = bool(c.persistent_workers)
            kwargs["prefetch_factor"] = int(c.prefetch_factor)
        return DataLoader(**kwargs)

    def train_dataloader(self):
        if self.train_set is None:
            raise RuntimeError("setup('fit') must be called before train_dataloader")
        return self._loader(self.train_set, train=True)

    def val_dataloader(self):
        if self.val_set is None:
            raise RuntimeError("setup('validate')/setup('fit') must be called before val_dataloader")
        return self._loader(self.val_set, train=False)
