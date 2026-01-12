# train_deep_models.py
import os
import random
import warnings
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics

import deep_models as models
from augmentation import Augment
import utils


# Label taxonomy (Walmsley 2020: 4 classes)

WALMSLEY_CLASSES = ["sleep", "sedentary", "light", "moderate-vigorous"]


def seed_worker(worker_id: int):
    """Safe top-level worker init function (picklable on Windows)."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ModelModule(pl.LightningModule):
    def __init__(self, model_cfg, optim_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg

        self.model = models.Resnet(
            model_cfg["n_channels"],
            model_cfg["outsize"],
            model_cfg["n_filters"],
            model_cfg["kernel_size"],
            model_cfg["n_resblocks"],
            model_cfg["resblock_kernel_size"],
            model_cfg["downfactor"],
            model_cfg["downorder"],
            model_cfg["drop1"],
            model_cfg["drop2"],
            model_cfg["fc_size"],
            model_cfg["is_cnnlstm"],
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # SWA
        self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
        self.register_buffer("is_swa_started", torch.tensor(False))

        # store outputs manually for Lightning v2
        self._val_outputs = []
        self._test_outputs = []

        # HMM params (buffers)
        n_classes = int(model_cfg["outsize"])
        self.register_buffer("hmm_prior", torch.zeros(n_classes))
        self.register_buffer("hmm_emission", torch.zeros(n_classes, n_classes))
        self.register_buffer("hmm_transition", torch.zeros(n_classes, n_classes))

        # safer matmul on tensor cores (RTX 3060 Ti)
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    def configure_optimizers(self):
        optim_cfg = self.optim_cfg

        if optim_cfg["method"] == "adam":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optim_cfg["adam"]["lr"],
                amsgrad=optim_cfg["adam"]["amsgrad"],
            )
            return optimizer

        if optim_cfg["method"] == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=optim_cfg["sgd"]["lr"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=optim_cfg["cosine_annealing"]["T_0"],
                T_mult=optim_cfg["cosine_annealing"]["T_mult"],
                eta_min=optim_cfg["cosine_annealing"]["eta_min"],
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        raise ValueError(f"Unknown optim method: {optim_cfg['method']}")

    def on_fit_start(self):
        # avoid computing len(train_dataloader) inside configure_optimizers (PL v2 issues)
        # use trainer-provided num batches once data is connected
        nb = getattr(self.trainer, "num_training_batches", None)
        if nb is None or (isinstance(nb, float) and (not np.isfinite(nb))):
            nb = 1
        self.num_batches_per_epoch = int(nb)

    def forward(self, x):
        if bool(self.is_swa_started):
            return self.swa_model(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        optimizer = self.optimizers()

        y = self.model(x)
        loss = self.loss_fn(y.view(-1, y.shape[-1]), target.view(-1))

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        # step scheduler if using SGD & before SWA
        if (not bool(self.is_swa_started)) and self.optim_cfg["method"] == "sgd":
            sched = self.lr_schedulers()
            # PL scheduler can be object or list; keep simple
            try:
                sched.step(self.global_step / max(self.num_batches_per_epoch, 1))
            except Exception:
                pass

        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # SWA start epoch
        start_epoch = int(self.optim_cfg["swa"]["start"])
        if (self.current_epoch + 1) >= start_epoch and (not bool(self.is_swa_started)):
            self.is_swa_started = torch.tensor(True, device=self.device)

        if bool(self.is_swa_started):
            self._update_swa_model()
            if self.optim_cfg["method"] == "sgd":
                self._adjust_learning_rate(float(self.optim_cfg["swa"]["lr"]))

    def validation_step(self, batch, batch_idx):
        out = self._eval_step(batch, tag="valid")
        self._val_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self._val_outputs
        if outputs:
            self._train_hmm(outputs)
            self._eval_epoch_end(outputs, tag="valid")
        self._val_outputs = []

    def test_step(self, batch, batch_idx):
        out = self._eval_step(batch, tag="test")
        self._test_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        outputs = self._test_outputs
        if outputs:
            self._eval_epoch_end(outputs, tag="test", print_report=True)
        self._test_outputs = []

    def _eval_step(self, batch, tag=""):
        x, target = batch
        y = self(x)
        loss = self.loss_fn(y.view(-1, y.shape[-1]), target.view(-1))
        self.log(f"{tag}/loss", loss, prog_bar=False)
        return y.detach(), target.detach()

    def _concat_outputs(self, outputs):
        Y_logit = torch.cat([o[0] for o in outputs], dim=0)
        Y_true = torch.cat([o[1] for o in outputs], dim=0)
        Y_logit = Y_logit.view(-1, Y_logit.shape[-1])
        Y_true = Y_true.view(-1)
        return Y_logit, Y_true

    def _eval_epoch_end(self, outputs, tag="", print_report=False):
        Y_logit, Y_true = self._concat_outputs(outputs)
        Y_pred = torch.argmax(Y_logit, dim=1)

        y_true = Y_true.cpu().numpy()
        y_pred = Y_pred.cpu().numpy()

        self._metrics_log(y_true, y_pred, tag=tag, print_report=print_report)

        # HMM smoothing (utils.viterbi expects class indices)
        prior = self.hmm_prior.cpu().numpy()
        emission = self.hmm_emission.cpu().numpy()
        transition = self.hmm_transition.cpu().numpy()

        y_pred_hmm = utils.viterbi(y_pred, prior, emission, transition)
        self._metrics_log(y_true, y_pred_hmm, tag=f"{tag}/hmm", print_report=print_report)

    def _train_hmm(self, outputs):
        # train on validation outputs (like original flow)
        Y_logit, Y_true = self._concat_outputs(outputs)
        y_pred = torch.argmax(Y_logit, dim=1).cpu().numpy()
        y_true = Y_true.cpu().numpy()

        # IMPORTANT: your utils.train_hmm returns a TUPLE: (prior, emission, transition)
        prior, emission, transition = utils.train_hmm(y_pred, y_true)

        self.hmm_prior = torch.as_tensor(prior, device=self.device, dtype=torch.float32)
        self.hmm_emission = torch.as_tensor(emission, device=self.device, dtype=torch.float32)
        self.hmm_transition = torch.as_tensor(transition, device=self.device, dtype=torch.float32)

    def _metrics_log(self, y_true, y_pred, tag="", print_report=False):
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0, average="macro")
        phi = metrics.matthews_corrcoef(y_true, y_pred)
        kappa = metrics.cohen_kappa_score(y_true, y_pred)

        self.log(f"{tag}/f1", f1, prog_bar=True)
        self.log(f"{tag}/phi", phi, prog_bar=False)
        self.log(f"{tag}/kappa", kappa, prog_bar=False)

        if print_report:
            # optional: keep it short; your utils.metrics_report may not exist
            try:
                print(f"\n[{tag}] f1={f1:.4f} phi={phi:.4f} kappa={kappa:.4f}\n")
            except Exception:
                pass

    def _update_swa_model(self):
        self.swa_model.update_parameters(self.model)

        # update BN stats for SWA model using train loader
        def _loader():
            train_loader = self.trainer.datamodule.train_dataloader()
            for batch in train_loader:
                x = batch[0].to(self.device)
                yield x

        with torch.no_grad():
            torch.optim.swa_utils.update_bn(_loader(), self.swa_model)

    def _adjust_learning_rate(self, lr: float):
        opt = self.optimizers()
        # PL returns optimizer directly when single, or wrapper
        if hasattr(opt, "optimizer"):
            opt = opt.optimizer
        for pg in opt.param_groups:
            pg["lr"] = lr


class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg, dataloader_cfg, augment_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.dataloader_cfg = dataloader_cfg
        self.augment_cfg = augment_cfg

        self.transform = Augment(
            augment_cfg["jitter"]["sigma"], augment_cfg["jitter"]["prob"],
            augment_cfg["shift"]["window"], augment_cfg["shift"]["prob"],
            augment_cfg["twarp"]["sigma"], augment_cfg["twarp"]["knots"], augment_cfg["twarp"]["prob"],
            augment_cfg["mwarp"]["sigma"], augment_cfg["mwarp"]["knots"], augment_cfg["mwarp"]["prob"],
        )

        self.dataset_train = None
        self.dataset_valid = None
        self.dataset_test = None

    def setup(self, stage=None):
        data_cfg = self.data_cfg
        datadir = os.path.expanduser(str(data_cfg["datadir"]))

        X = np.load(os.path.join(datadir, "X.npy"), mmap_mode="r")
        pid = np.load(os.path.join(datadir, "pid.npy"))

        # FORCE Walmsley-4 baseline labels
        Y = np.load(os.path.join(datadir, "Y_Walmsley2020.npy"))

        # ensure utils matches 4-class taxonomy
        utils.CLASSES = WALMSLEY_CLASSES
        utils.NUM_CLASSES = len(utils.CLASSES)
        utils.CLASS_CODE = {c: i for i, c in enumerate(utils.CLASSES)}
        utils.COLORS = ["blue", "red", "darkorange", "green"]  # optional

        # deriv/test split: P001-P100 for derivation, rest for test
        whr_deriv = np.isin(pid, [f"P{i:03d}" for i in range(1, 101)])
        X_deriv, Y_deriv, pid_deriv = X[whr_deriv], Y[whr_deriv], pid[whr_deriv]
        X_test, Y_test, pid_test = X[~whr_deriv], Y[~whr_deriv], pid[~whr_deriv]

        # split deriv into train/val by pid
        rng = np.random.default_rng(int(data_cfg["seed"]) if "seed" in data_cfg else 42)
        unique_pids = np.unique(pid_deriv)
        val_size = int(data_cfg["val_size"])
        val_pids = rng.choice(unique_pids, size=val_size, replace=False)
        whr_val = np.isin(pid_deriv, val_pids)

        X_val, Y_val = X_deriv[whr_val], Y_deriv[whr_val]
        X_train, Y_train = X_deriv[~whr_val], Y_deriv[~whr_val]

        self.dataset_train = Dataset(X_train, Y_train, transform=self.transform, seq_length=int(data_cfg["seq_length"]))
        self.dataset_valid = Dataset(X_val, Y_val, transform=None, seq_length=int(data_cfg["seq_length"]))
        self.dataset_test = Dataset(X_test, Y_test, transform=None, seq_length=int(data_cfg["seq_length"]))

    def train_dataloader(self):
        return self._create_dataloader(self.dataset_train, self.dataloader_cfg["train"], deterministic=False)

    def val_dataloader(self):
        return self._create_dataloader(self.dataset_valid, self.dataloader_cfg["valid"], deterministic=True)

    def test_dataloader(self):
        return self._create_dataloader(self.dataset_test, self.dataloader_cfg["test"], deterministic=True)

    @staticmethod
    def _create_dataloader(dataset, cfg, deterministic=False):
        batch_size = int(cfg.get("batch_size", 64))
        num_workers = int(cfg.get("num_workers", 0))  # keep 0 on Windows to avoid spawn issues
        seed = int(cfg.get("seed", 12345))

        if deterministic and num_workers > 0:
            warnings.warn("Deterministic + num_workers>0 can be non-reproducible on some Windows setups.")

        g = torch.Generator()
        g.manual_seed(seed)

        return DataLoader(
            dataset,
            shuffle=(not deterministic),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=g,
            persistent_workers=(num_workers > 0),
        )


class Dataset(torch.utils.data.Dataset):
    """
    X windows are expected as (window_len, n_channels)
    We return torch tensor as (n_channels, window_len) for Conv1d.
    Labels must be strings in WALMSLEY_CLASSES.
    """

    def __init__(self, X, Y=None, transform=None, seq_length=1):
        self.seq_length = int(seq_length)
        self.transform = transform

        if self.seq_length > 1:
            nX = int((len(X) // self.seq_length) * self.seq_length)
            X = [X[i:i + self.seq_length] for i in range(0, nX, self.seq_length)]
            if Y is not None:
                Y = [Y[i:i + self.seq_length] for i in range(0, nX, self.seq_length)]

        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = self.prepare_x(x)

        if self.Y is None:
            return x

        y = self.Y[idx]
        y = self.prepare_y(y)
        return x, y

    @staticmethod
    def to_class_idx(y):
        # y is string label
        y = str(y)
        return WALMSLEY_CLASSES.index(y)

    def _prepare_x_single(self, x):
        # x: (T, C)
        if self.transform is not None:
            x = self.transform(x)
        x = np.asarray(x).T  # (C, T)
        return torch.tensor(x, dtype=torch.float32)

    def prepare_x(self, x):
        if self.seq_length > 1:
            # return (seq, C, T)
            return torch.stack([self._prepare_x_single(_x) for _x in x], dim=0)
        return self._prepare_x_single(x)

    def prepare_y(self, y):
        if self.seq_length > 1:
            return torch.tensor([Dataset.to_class_idx(_y) for _y in y], dtype=torch.long)
        return torch.tensor(Dataset.to_class_idx(y), dtype=torch.long)


def resolve_cfg_paths(cfg: DictConfig):
    cfg.data.datadir = os.path.expanduser(cfg.data.datadir)
    if cfg.ckpt_path is not None:
        cfg.ckpt_path = os.path.expanduser(cfg.ckpt_path)


def create_lightning_modules(cfg: DictConfig):
    datamodule = DataModule(cfg.data, cfg.dataloader, cfg.augment)
    if cfg.ckpt_path is not None:
        model = ModelModule.load_from_checkpoint(cfg.ckpt_path)
    else:
        model = ModelModule(cfg.model, cfg.optim)
    return datamodule, model


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(int(cfg.seed), workers=True)
    resolve_cfg_paths(cfg)

    datamodule, model = create_lightning_modules(cfg)

    early_stop_callback = EarlyStopping(monitor="valid/loss", patience=int(cfg.early_stop_patience), mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="valid/loss", mode="min", filename="best", save_top_k=1)

    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision="16-mixed" if use_gpu else 32,
        max_epochs=int(cfg.n_epochs),
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        limit_test_batches=cfg.limit_test_batches,
        deterministic=True,
        enable_checkpointing=True,
    )

    if bool(cfg.fit):
        trainer.fit(model, datamodule=datamodule)

    if bool(cfg.test):
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
