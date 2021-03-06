# import cv2
import numpy as np
# import h5py
from math import floor
import torch.utils.data.dataloader
from torch.optim import lr_scheduler
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from aermanager.preprocess import accumulate_frames, slice_by_time
# from aermanager.cvat_dataset_generator import load_rois_lut, load_annotated_slice
# from tqdm import tqdm

class CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.train_dataloader = dataloader_train
        # self.val_dataloader = dataloader_val
        self.network = nn.Sequential(
            # 
            nn.Conv2d(8, 16, (8, 1), 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(16, 16, (10, 1), 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(3264, 12, bias=False)
        )

    def forward(self, x):
        output = self.network(x)
        return output

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def training_step(self, batch, batch_idx):
        frame, target = batch
        data, target = frame.float(), target
        output = self.network(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / (len(target))
        tensorboard_logs = {'train_loss': loss, 'train_acc': accuracy}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        frame, target = batch
        data, target = frame.float(), target
        output = self.network(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / (len(target))
        return {'val_loss': F.cross_entropy(output, target), 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc_l = [x['val_acc'] for x in outputs]
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': sum(acc_l) / len(acc_l)}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-5, eps=1e-8)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'monitor': 'val_loss', 'interval': 'epoch', 'scheduler': scheduler}

    def on_train_end(self):
        torch.save(self.network.state_dict(), "shapes_weights.pt")

    def test_step(self, batch, batch_nb):
        frame, target = batch
        data, target = frame.float(), target
        output = self.network(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / (len(target))
        return {'test_loss': F.cross_entropy(output, target), 'test_acc': accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        acc_l = [x['test_acc'] for x in outputs]
        logs = {'test_loss': avg_loss, 'test_acc': sum(acc_l) / len(acc_l)}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}