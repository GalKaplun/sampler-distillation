import os

import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax
from torchvision import models


def modified_res18():
    resnet = models.resnet18(pretrained=False, num_classes=10)
    resnet.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    resnet.maxpool = torch.nn.Identity()
    return resnet


def get_model(model_name):
    if model_name.lower() == 'resnet18':
        return modified_res18()
    else:
        return models.__dict__[model_name.lower()](pretrained=False, num_classes=10)

def get_teachers(teach_arch, teach_dir, max_teachers):
    from glob import glob
    teachers = []
    for i, teacher_checkpoint in enumerate(glob(os.path.join(teach_dir, '*', 'last.ckpt'))):
        if i >= max_teachers:
            break
        teacher = get_model(teach_arch)
        state_dict = torch.load(teacher_checkpoint)["state_dict"]
        state_dict = {k[6:]: v for k, v in state_dict.items()}
        teacher.load_state_dict(state_dict)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.cuda()
        teachers += [teacher]
    return teachers


class Cifar10Model(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        learning_rate: float = 1e-1,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        schedule: str = 'step',
        teach_arch: str = 'non',
        teach_dir: str = '',
        hard_labels: bool = True,
        ensemble: bool = False,
        max_teachers: int = 1,
        dataset: str = 'cifar10'
    ):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = get_model(arch)
        if teach_arch.lower() != 'non':
            self.teachers = get_teachers(teach_arch, teach_dir, max_teachers)
        else:
            self.teachers = None
        self.hard_labels = hard_labels
        self.ensemble = ensemble

        self.train_acc = Accuracy()
        self.pred_accs = Accuracy()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.is_cifar5m = dataset.lower() == 'cifar5m'
        if self.is_cifar5m:
            self.max_epochs *= 100

        self.schedule = schedule

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        images, _ = batch
        return self.model(images)

    def process_batch(self, batch, stage="train"):
        images, labels = batch
        if stage == "train" and self.teachers is not None:
            if self.ensemble:
                labels = None
                with torch.no_grad():
                    for curr_teacher in self.teachers:
                        if labels is None:
                            labels = curr_teacher(images)
                        else:
                            labels += curr_teacher(images)
                    if self.hard_labels:
                        labels = torch.argmax(labels, dim=1)
            else:
                perm = torch.randperm(len(self.teachers))
                idx = perm[0]
                curr_teacher = self.teachers[idx]
                with torch.no_grad():
                    labels = curr_teacher(images)
                    if self.hard_labels:
                        labels = torch.argmax(labels, dim=1)
        logits = self.forward(images)
        probs = softmax(logits, dim=1)
        loss = self.criterion(logits, labels)

        if stage == "train":
            self.train_acc(probs, labels)
        elif stage == "pred":
            self.pred_accs(probs, labels)
        else:
            raise ValueError("Invalid stage %s" % stage)

        return loss

    def training_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "train")
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)

        # a bit hacky
        if self.is_cifar5m and batch_idx % 400 == 399:
            sch = self.lr_schedulers()
            sch.step()
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "pred")
        self.log("pred_loss", loss)
        self.log(
            "pred_acc",
            self.pred_accs,
            on_step=False or self.is_cifar5m,
            on_epoch=True
        )

    def configure_optimizers(self):
        parameters = self.model.parameters()

        optimizer = torch.optim.SGD(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9
        )

        if self.schedule == 'step':
            lr_scheduler = lr_sched.StepLR(optimizer, step_size=self.max_epochs//3 + 1, gamma=0.1)
        elif self.schedule == 'cos':
            lr_scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        else:
            raise NotImplementedError()

        return [optimizer], [lr_scheduler]
