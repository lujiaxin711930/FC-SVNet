# -*- coding: utf-8 -*-
"""
train.py
Converted from the original Kaggle notebook into a standalone Python script.
Preserves the original notebook workflow.
"""

import math
import os
import time
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision  # noqa: F401  (Keep the original import to preserve the intended code structure)
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Global configuration: update the paths as needed
# ============================================================
DATA_DIR = 'No_deep_Data_hua'      # Dataset root containing train/ and val/ splits, each with class folders 0 through 6
OUTPUT_DIR = 'outputs'             # Directory for saved models, checkpoints, and training plots
NUM_CLASSES = 7


# ========== Set random seeds ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


# ========== WellSequenceDataset: group by well and depth; provide y_prev and y_next ==========
class WellSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # (img_path, label, y_prev, y_next, well_id)

        all_files = []
        for label in range(7):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                name_no_ext = fname.rsplit('.', 1)[0]
                parts = name_no_ext.split('_')
                if len(parts) < 3:
                    continue
                well_id = parts[0]
                depth = float(parts[1])
                count = int(parts[2])
                img_path = os.path.join(class_dir, fname)
                all_files.append((well_id, depth, count, img_path, label))

        well_groups = defaultdict(list)
        for well_id, depth, count, img_path, label in all_files:
            well_groups[well_id].append((depth, count, img_path, label))

        for well_id, group in well_groups.items():
            group.sort(key=lambda x: (x[0], x[1]))
            n = len(group)
            for i in range(n):
                depth, count, img_path, label = group[i]
                y_prev = group[i - 1][3] if i > 0 else -1
                y_next = group[i + 1][3] if i < n - 1 else -1
                self.samples.append((img_path, label, y_prev, y_next, well_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, y_prev, y_next, well_id = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, y_prev, y_next, well_id


# ========== Batch collation function ==========
def collate_fn(batch):
    images, labels, y_prevs, y_nexts, well_ids = zip(*batch)
    return (
        torch.stack(images),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(y_prevs, dtype=torch.long),
        torch.tensor(y_nexts, dtype=torch.long),
        well_ids
    )


# ========== Mona adapter module ==========
class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3,
                               padding=1, groups=in_features, bias=False)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5,
                               padding=2, groups=in_features, bias=False)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7,
                               padding=3, groups=in_features, bias=False)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_features)

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        identity = x
        x = self.projector(x)
        x = self.bn(x)
        return identity + x


class MonaAdapter(nn.Module):
    def __init__(self, in_dim, inner_dim=64, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.inner_dim = inner_dim
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.beta = nn.Parameter(torch.ones(in_dim))
        self.down_proj = nn.Linear(in_dim, inner_dim)
        self.adapter_conv = MonaOp(inner_dim)
        self.up_proj = nn.Linear(inner_dim, in_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat) * self.gamma + x_flat * self.beta
        x_down = self.down_proj(x_norm)
        x_2d = x_down.reshape(B, H, W, self.inner_dim).permute(0, 3, 1, 2)
        x_conv = self.adapter_conv(x_2d)
        x_conv_flat = x_conv.permute(0, 2, 3, 1).reshape(B, H * W, self.inner_dim)
        x_act = self.act(x_conv_flat)
        x_act = self.dropout(x_act)
        x_up = self.up_proj(x_act)
        out = x_up.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return identity + out


# ========== Model building blocks ==========
class SPBlock(nn.Module):
    def __init__(self, inplanes):
        super(SPBlock, self).__init__()
        outplanes = inplanes
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(outplanes, outplanes, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        x = self.relu(x1 + x2)
        x = self.conv3(x).sigmoid()
        return x


class MGCBlock(nn.Module):
    def __init__(self, inplans, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(MGCBlock, self).__init__()
        outplanes = inplans
        self.conv_1 = nn.Conv2d(inplans, outplanes // 4, kernel_size=conv_kernels[0],
                                padding=conv_kernels[0] // 2, stride=stride, groups=conv_groups[0])
        self.conv_2 = nn.Conv2d(inplans, outplanes // 4, kernel_size=conv_kernels[1],
                                padding=conv_kernels[1] // 2, stride=stride, groups=conv_groups[1])
        self.conv_3 = nn.Conv2d(inplans, outplanes // 4, kernel_size=conv_kernels[2],
                                padding=conv_kernels[2] // 2, stride=stride, groups=conv_groups[2])
        self.conv_4 = nn.Conv2d(inplans, outplanes // 4, kernel_size=conv_kernels[3],
                                padding=conv_kernels[3] // 2, stride=stride, groups=conv_groups[3])

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        return out


class SMGC(nn.Module):
    def __init__(self, inplanes):
        super(SMGC, self).__init__()
        self.sp = SPBlock(inplanes)
        self.mgc = MGCBlock(inplanes)
        self.conv1 = nn.Conv2d(inplanes, inplanes, 1)
        self.conv2 = nn.Conv2d(inplanes * 3, inplanes, 1)

    def forward(self, x):
        x1 = self.sp(x)
        x2 = self.mgc(x)
        x3 = self.conv1(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.conv2(out)
        return out


class ECAs(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECAs, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y = torch.add(y1, y2)
        y = torch.sigmoid(y)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ResNet50BasicBlock(nn.Module):
    def __init__(self, in_channel, outs, kernerl_size, stride, padding):
        super(ResNet50BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernerl_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernerl_size[1], stride=stride[0], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernerl_size[2], stride=stride[0], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        out = self.bn3(out)
        return F.relu(out + x)


class ResNet50DownBlock(nn.Module):
    def __init__(self, in_channel, outs, kernel_size, stride, padding):
        super(ResNet50DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(outs[2])
        self.extra = nn.Sequential(
            nn.Conv2d(in_channel, outs[2], kernel_size=1, stride=stride[3], padding=0),
            nn.BatchNorm2d(outs[2])
        )

    def forward(self, x):
        x_shortcut = self.extra(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        return F.relu(x_shortcut + out)


# ========== ResNet50 with a learnable facies transition matrix ==========
class ResNet50(nn.Module):
    def __init__(self, trans_init=None):
        super(ResNet50, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ResNet50DownBlock(64, outs=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            SMGC(256)
        )
        self.layer2 = nn.Sequential(
            ResNet50DownBlock(256, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            SMGC(512)
        )
        self.layer3 = nn.Sequential(
            ResNet50DownBlock(512, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            SMGC(1024)
        )
        self.layer4 = nn.Sequential(
            ResNet50DownBlock(1024, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(2048, outs=[512, 512, 2048], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(2048, outs=[512, 512, 2048], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            SMGC(2048),
            ECAs(2048),
        )
        self.mona_adapter = MonaAdapter(in_dim=2048, inner_dim=64, dropout=0.1)

        # Branch 1: spatial features
        self.fc_spatial = nn.Linear(2048, 256)

        # Branch 2: GRU sequence features
        self.gru_input_dim = 2048
        self.gru_hidden_dim = 256
        self.gru_num_layers = 1
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=self.gru_num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc_seq = nn.Linear(self.gru_hidden_dim * 2, 256)

        # Learnable gating coefficient
        self.gate = nn.Parameter(torch.tensor(0.3))

        # Shared classification head
        self.drop = torch.nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256, NUM_CLASSES)

        # Learnable lithofacies transition matrix
        if trans_init is not None:
            self.trans_matrix = nn.Parameter(trans_init.clone())
        else:
            self.trans_matrix = nn.Parameter(torch.zeros(NUM_CLASSES, NUM_CLASSES))

    def forward(self, x, y_prev=None, y_next=None):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu0(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.mona_adapter(out)

        B, C, H, W = out.shape

        # Branch 1: spatial features
        spatial_pooled = F.adaptive_avg_pool2d(out, (1, 1))
        spatial_pooled = spatial_pooled.view(B, -1)
        spatial_feat = self.fc_spatial(spatial_pooled)

        # Branch 2: sequence features
        seq_pooled = F.adaptive_avg_pool2d(out, (H, 1))
        seq_pooled = seq_pooled.squeeze(-1)
        gru_input = seq_pooled.permute(0, 2, 1)
        gru_out, _ = self.gru(gru_input)
        gru_final = torch.mean(gru_out, dim=1)
        seq_feat = self.fc_seq(gru_final)

        # Gated feature fusion
        gate_weight = torch.sigmoid(self.gate)
        fused_feat = (1 - gate_weight) * spatial_feat + gate_weight * seq_feat

        # Classification head
        out = self.drop(fused_feat)
        out = self.fc1(out)
        return out


# ========== Geological sequence consistency loss ==========
def geological_sequence_loss(logits, y_prev, y_next, trans_matrix, temperature=1.0):
    B, C = logits.shape
    total_loss = 0.0
    count = 0

    if y_prev is not None:
        mask = y_prev >= 0
        if mask.any():
            trans_scores = trans_matrix[y_prev[mask]]
            trans_dist = F.softmax(trans_scores / temperature, dim=1)
            kl_loss = F.kl_div(
                F.log_softmax(logits[mask] / temperature, dim=1),
                trans_dist,
                reduction='sum',
                log_target=False
            )
            total_loss += kl_loss
            count += mask.sum().item()

    if y_next is not None:
        mask = y_next >= 0
        if mask.any():
            trans_scores = trans_matrix[:, y_next[mask]].t()
            trans_dist = F.softmax(trans_scores / temperature, dim=1)
            kl_loss = F.kl_div(
                F.log_softmax(logits[mask] / temperature, dim=1),
                trans_dist,
                reduction='sum',
                log_target=False
            )
            total_loss += kl_loss
            count += mask.sum().item()

    if count > 0:
        return total_loss / count
    else:
        return torch.tensor(0.0, device=logits.device)


# ========== ProgressiveFreezing class ==========
class ProgressiveFreezing:
    def __init__(self, patience=5, delta=0.001, min_epochs=3, max_epochs=15):
        self.patience = patience
        self.delta = delta
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.counter = 0
        self.best_acc = None
        self.frozen = False
        self.freeze_epoch = 0
        self.trigger_reason = ""

    def __call__(self, val_acc, current_epoch):
        if self.frozen or current_epoch < self.min_epochs:
            return False
        if current_epoch >= self.max_epochs:
            self.frozen = True
            self.trigger_reason = f"Maximum epoch limit reached({self.max_epochs}epochs)"
            return True
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc <= self.best_acc + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.frozen = True
                self.trigger_reason = f"For{self.patience}consecutive epochs, validation accuracy improved by less than{self.delta * 100}%"
                return True
        else:
            self.best_acc = val_acc
            self.counter = 0
        return False

    def freeze_backbone(self, model, optimizer):
        for name, param in model.named_parameters():
            if not any(key in name for key in
                       ['mona_adapter', 'fc_spatial', 'fc_seq', 'gate', 'gru', 'fc1', 'trans_matrix']):
                param.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer.__init__(trainable_params, lr=5e-4, weight_decay=0.01)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


# ========== Checkpoint save/resume utilities ==========
def save_checkpoint(filepath, epoch, model, optimizer, scheduler,
                    best_val_acc, best_val_loss, best_epoch,
                    train_loss, train_acc, test_loss, test_acc, epoch_sum,
                    is_frozen, progressive_freezing):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'epoch_sum': epoch_sum,
        'is_frozen': is_frozen,
        'progressive_freezing_counter': progressive_freezing.counter,
        'progressive_freezing_best_acc': progressive_freezing.best_acc,
        'progressive_freezing_frozen': progressive_freezing.frozen,
        'progressive_freezing_freeze_epoch': progressive_freezing.freeze_epoch,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath} (epoch {epoch})")


def load_checkpoint(filepath, model, optimizer, scheduler, progressive_freezing):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    train_loss = checkpoint['train_loss']
    train_acc = checkpoint['train_acc']
    test_loss = checkpoint['test_loss']
    test_acc = checkpoint['test_acc']
    epoch_sum = checkpoint['epoch_sum']
    is_frozen = checkpoint['is_frozen']

    progressive_freezing.counter = checkpoint['progressive_freezing_counter']
    progressive_freezing.best_acc = checkpoint['progressive_freezing_best_acc']
    progressive_freezing.frozen = checkpoint['progressive_freezing_frozen']
    progressive_freezing.freeze_epoch = checkpoint['progressive_freezing_freeze_epoch']

    print(f"Resumed from checkpoint: epoch {epoch}, best_val_acc {best_val_acc:.4f}, is_frozen {is_frozen}")
    return (epoch, best_val_acc, best_val_loss, best_epoch,
            train_loss, train_acc, test_loss, test_acc, epoch_sum,
            is_frozen)


# ============================================================
# Main training workflow
# ============================================================
def main():
    set_seed(42)

    # ========== Device configuration ==========
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== Image transforms ==========
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(p=0.03),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])
    }

    # ========== Dataset and data loader initialization ==========
    image_datasets = {x: WellSequenceDataset(
        root_dir=os.path.join(DATA_DIR, x),
        transform=data_transform[x]
    ) for x in ['train', 'val']}

    dataloader = {x: torch.utils.data.DataLoader(
        dataset=image_datasets[x],
        batch_size=64,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    ) for x in ['train', 'val']}

    # ========== Estimate the lithofacies transition prior ==========
    num_classes = NUM_CLASSES
    trans_counts = np.ones((num_classes, num_classes))  # Laplace smoothing

    for _, label, y_prev, y_next, _ in image_datasets['train'].samples:
        if y_prev >= 0:
            trans_counts[y_prev, label] += 1
        if y_next >= 0:
            trans_counts[label, y_next] += 1

    trans_prob = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    trans_init = torch.log(torch.from_numpy(trans_prob).float() + 1e-8)
    print(f"\n{'=' * 60}")
    print("Lithofacies transition prior estimated")
    print(f"Transition matrix shape: {trans_init.shape}")
    print(f"Transition probability matrix (7x7):\n{np.round(trans_prob, 3)}")
    print(f"{'=' * 60}\n")

    # ========== Model initialization ==========
    model = ResNet50(trans_init=trans_init).to(device)
    print(model)

    # ========== Loss function and optimizer ==========
    loss_f = torch.nn.CrossEntropyLoss()

    backbone_params = []
    adapter_params = []

    for name, param in model.named_parameters():
        if any(key in name for key in
               ['mona_adapter', 'fc_spatial', 'fc_seq', 'gate', 'gru', 'fc1', 'trans_matrix']):
            adapter_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4, 'weight_decay': 0.01},
        {'params': adapter_params, 'lr': 5e-4, 'weight_decay': 0.01},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # ========== Training loop ==========
    epoch_n = 100
    time_open = time.time()
    train_loss, train_acc, test_loss, test_acc, epoch_sum = [], [], [], [], []
    best_val_acc, best_val_loss, best_epoch = 0.0, float('inf'), 0

    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 60}")
    print("【Stage 1】Start full-parameter training")
    print(f"Total parameters: {total:,} (100% trainable)")
    print(f"  - backbone parameters (lr=1e-4): {sum(p.numel() for p in backbone_params):,}")
    print(f"  - adapter + transition matrix parameters (lr=5e-4): {sum(p.numel() for p in adapter_params):,}")
    print(f"{'=' * 60}\n")

    progressive_freezing = ProgressiveFreezing(patience=5, delta=0.001, max_epochs=15)
    is_frozen = False

    # ========== Check whether a checkpoint exists ==========
    checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoint.pth')

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found; resuming...")
        (resume_epoch, best_val_acc, best_val_loss, best_epoch,
         train_loss, train_acc, test_loss, test_acc, epoch_sum,
         is_frozen) = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, progressive_freezing
        )
        print(f"Resuming at epoch {resume_epoch + 1}")
    else:
        print("No checkpoint found; training from scratch")
        resume_epoch = 0
        best_val_acc, best_val_loss, best_epoch = 0.0, float('inf'), 0
        train_loss, train_acc, test_loss, test_acc, epoch_sum = [], [], [], [], []
        is_frozen = False

    for epoch in range(resume_epoch, epoch_n):
        print(f'epoch {epoch}/{epoch_n - 1}')
        print('-' * 10)

        if not is_frozen:
            print("【Stage 1】Full-parameter training (backbone LR=1e-4; adapter and transition matrix LR=5e-4)")
        else:
            print(f"【Stage 2】Freeze backbone; optimize only Mona+GRU+Classification head+transition matrix (epoch {epoch - progressive_freezing.freeze_epoch} after freezing)")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0

            for batch, data in enumerate(dataloader[phase], 1):
                X, Y, y_prev, y_next, well_ids = data
                X = X.to(device)
                Y = Y.to(device)
                y_prev = y_prev.to(device)
                y_next = y_next.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X)
                    _, pred = torch.max(y_pred, 1)

                    # 1. Cross-entropy loss
                    ce_loss = loss_f(y_pred, Y)

                    # 2. Sequence consistency loss
                    seq_loss = geological_sequence_loss(
                        y_pred,
                        y_prev,
                        y_next,
                        model.trans_matrix,
                        temperature=1.0
                    )

                    # 3. Total loss
                    lambda_seq = 0.3
                    loss = ce_loss + lambda_seq * seq_loss

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * X.size(0)
                running_corrects += torch.sum(pred == Y.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f'{phase} Loss:{epoch_loss:.4f} Acc:{epoch_acc:.4f}')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.item())
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc.item())

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_val_loss = epoch_loss
                    best_epoch = epoch + 1
                    torch.save(model, os.path.join(OUTPUT_DIR, 'best_model_FC-SVNet.pth'))
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_params.pth'))
                    print(f"New best model saved; best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

                if not is_frozen:
                    should_freeze = progressive_freezing(epoch_acc, epoch)
                    if should_freeze:
                        is_frozen = True
                        progressive_freezing.freeze_epoch = epoch
                        scheduler = progressive_freezing.freeze_backbone(model, optimizer)

                        total = sum(p.numel() for p in model.parameters())
                        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        print(f"\n{'=' * 60}")
                        print(f"【Progressive freezing triggered】epoch {epoch}epochs，{progressive_freezing.trigger_reason}")
                        print(f"Freezing backbone parameters（ResNet+SMGC+ECAs）")
                        print(f"Optimizing only Mona+GRU+classification head+transition matrix")
                        print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
                        print(f"{'=' * 60}\n")

        scheduler.step()
        epoch_sum.append(epoch)

        save_checkpoint(
            checkpoint_path, epoch, model, optimizer, scheduler,
            best_val_acc, best_val_loss, best_epoch,
            train_loss, train_acc, test_loss, test_acc, epoch_sum,
            is_frozen, progressive_freezing
        )

    # ========== Save final model ==========
    torch.save(model, os.path.join(OUTPUT_DIR, 'final_model_FC-SVNet.pth'))
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model_params.pth'))
    print("\n" + "=" * 60)
    print("Training complete - model save details")
    print(f"Best model: validation accuracy {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Final model: saved after {epoch_n} epochs")
    print("=" * 60 + "\n")

    print("get train_loss_min:{} at epoch:{}".format(min(train_loss), train_loss.index(min(train_loss)) + 1))
    print('get train_acc_max:{} at epoch:{}'.format(max(train_acc), train_acc.index(max(train_acc)) + 1))
    print('get val_loss_min:{} at epoch:{}'.format(min(test_loss), test_loss.index(min(test_loss)) + 1))
    print("get val_acc_max:{} at epoch:{}".format(max(test_acc), test_acc.index(max(test_acc)) + 1))
    print(f"Total training time: {time.time() - time_open:.2f} seconds")

    # ========== Save loss and accuracy history ==========
    metrics_data = {
        'epoch': epoch_sum,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': test_loss,
        'val_acc': test_acc
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'train_val_metrics.csv'), index=False, encoding='utf-8')
    np.save(os.path.join(OUTPUT_DIR, 'train_val_metrics.npy'), metrics_data)
    print("=" * 30 + " Data saved " + "=" * 30)

    # ========== Plot results ==========
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_sum, train_loss, 'y-', label='train_loss')
    plt.plot(epoch_sum, test_loss, 'g-', label='val_loss')
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epochs', fontsize=20)
    plt.legend(fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss.png'), dpi=600, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_sum, train_acc, 'r-', label='train_acc')
    plt.plot(epoch_sum, test_acc, 'b-', label='val_acc')
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Epochs', fontsize=20)
    plt.legend(fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, 'acc.png'), dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()