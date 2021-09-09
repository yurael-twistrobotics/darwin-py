from functools import partial
from typing import Any, Dict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import AverageMeter
from torchmetrics.metric import Metric


class Classifier(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        metrics: Metric,
        lr: float = 0.0002,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_metrics = metrics
        self.val_metrics = metrics.clone()
        self.train_avg_loss = AverageMeter()
        self.val_avg_loss = AverageMeter()
        self.lr = lr

    def namespaced(self, name: str, metrics: Dict[str, Any]):
        return {f"{name}_{k}": v for k, v in metrics.items()}

    def step(self, batch, batch_idx, metrics):
        x, y = batch
        outs = self.model(x)
        loss = self.criterion(outs, y)
        preds = F.softmax(outs, dim=1)
        return {"loss": loss, "preds": preds.detach(), "target": y.detach()}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.train_metrics)

    def training_step_end(self, outputs):
        self.train_metrics(outputs["preds"], outputs["target"])
        self.train_avg_loss.update(outputs["loss"])
        self.log_dict(self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.val_metrics)

    def validation_step_end(self, outputs):
        self.val_metrics(outputs["preds"], outputs["target"])
        self.val_avg_loss.update(outputs["loss"])

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.val_metrics)

    def test_step_end(self, outputs):
        self.val_metrics(outputs["preds"], outputs["target"])
        self.val_avg_loss.update(outputs["loss"])

    def default_on_epoch_end(self, namespace: str, metrics: Metric, avg_loss: Metric):
        computed = metrics.compute()
        computed = {"loss": avg_loss.compute(), **computed}
        computed = self.namespaced(f"{namespace}_avg", computed)
        self.log_dict(computed)
        metrics.reset()
        avg_loss.reset()

    def on_train_epoch_end(self) -> None:
        self.default_on_epoch_end("train", self.train_metrics, self.train_avg_loss)

    def on_validation_epoch_end(self) -> None:
        self.default_on_epoch_end("val", self.val_metrics, self.val_avg_loss)

    def on_test_epoch_end(self) -> None:
        self.default_on_epoch_end("test", self.val_metrics, self.val_avg_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


SingleLabelClassifier = partial(Classifier, criterion=torch.nn.CrossEntropyLoss)
MultiLabelClassifier = partial(Classifier, criterion=torch.nn.BCEWithLogitsLoss)
