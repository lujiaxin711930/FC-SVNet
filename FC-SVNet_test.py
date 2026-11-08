# -*- coding: utf-8 -*-
"""
test.py
Converted from the original Kaggle notebook into a standalone Python script.
Implements independent predictions and Viterbi sequence decoding with a fixed alpha.
"""

import math
import os
import time

import numpy as np
import torch
import torchvision  # noqa: F401
from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# Path configuration: update as needed
# ============================================================
DATA_DIR = 'No_deep_Data_hua'          # Dataset root containing test/0 through test/6
OUTPUT_DIR = 'outputs'                  # Directory for test outputs
MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model_FC-SVNet.pth')  # Serialized full model
PARAMS_PATH = os.path.join(OUTPUT_DIR, 'best_model_params.pth')                     # Model weights

os.makedirs(OUTPUT_DIR, exist_ok=True)

num_classes = 7  # Explicitly define the number of classes for standalone execution

# Fixed alpha used for Viterbi decoding
SELECTED_ALPHA = 0.40

# ========== 1. Matplotlib backend ==========
# Use the Agg backend for headless execution; comment out this line to display plots interactively
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# ========== Model architecture matching the training script ==========
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
        self.fc1 = nn.Linear(256, 7)

        # Learnable lithofacies transition matrix
        if trans_init is not None:
            self.trans_matrix = nn.Parameter(trans_init.clone())
        else:
            self.trans_matrix = nn.Parameter(torch.zeros(num_classes, num_classes))

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


# ========== Viterbi decoding ==========
def viterbi_decode(logits_list, trans_log, alpha=SELECTED_ALPHA):
    """
    Viterbi decoding, where alpha controls the influence of transition scores.
    alpha=0: ignore transition scores (independent predictions).
    alpha=1: apply the full transition score contribution.
    """
    n = len(logits_list)
    if n == 0:
        return []

    log_probs = torch.stack([F.log_softmax(l, dim=0) for l in logits_list])
    dp = torch.zeros(n, 7)
    path = torch.zeros(n, 7, dtype=torch.long)

    dp[0] = log_probs[0]

    for i in range(1, n):
        for j in range(7):
            scores = dp[i - 1] + alpha * trans_log[:, j] + log_probs[i, j]
            dp[i, j], path[i, j] = scores.max(dim=0)

    best_last = dp[-1].argmax().item()
    best_path = [best_last]
    for i in range(n - 1, 0, -1):
        best_path.append(path[i, best_path[-1]].item())

    return best_path[::-1]


# ========== Generate and save confusion matrices and metrics ==========
def generate_and_save_metrics(preds, labels, alpha, acc, prefix='Viterbi'):
    """
    Compute confusion matrices and metrics, saving figures and text reports.
    Use the same plot style as confusion_matrix_single.
    """
    classes = ['St', 'Sa', 'Sp', 'Sw', 'Fw', 'Fh', 'O']
    C = confusion_matrix(
        [str(i) for i in labels],
        [str(i) for i in preds]
    )

    # ---------- Count-based confusion matrix ----------
    plt.figure(figsize=(8, 6))
    plt.matshow(C, cmap=plt.cm.Greens, fignum=None)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.colorbar()
    plt.ylabel('True label', size=12)
    plt.xlabel('Predicted label', size=12)
    plt.xticks(np.arange(7), labels=classes)
    plt.yticks(np.arange(7), labels=classes)
    plt.title(f'Confusion matrix ({prefix}) alpha={alpha:.2f}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{prefix}_alpha{alpha:.2f}.png'),
                dpi=600, bbox_inches='tight')
    plt.close()

    # ---------- Normalized confusion matrix ----------
    C_norm = C.astype(np.float64)
    row_sums = C_norm.sum(axis=1)
    for i in range(len(C_norm)):
        for j in range(len(C_norm)):
            C_norm[i][j] = np.round(((C_norm[i][j] / row_sums[i]) * 100), 1) if row_sums[i] != 0 else 0

    plt.figure(figsize=(8, 6))
    plt.matshow(C_norm, cmap=plt.cm.Greens, fignum=None)
    for i in range(len(C_norm)):
        for j in range(len(C_norm)):
            plt.annotate(C_norm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.colorbar()
    plt.ylabel('True label', size=12)
    plt.xlabel('Predicted label', size=12)
    plt.xticks(np.arange(7), labels=classes)
    plt.yticks(np.arange(7), labels=classes)
    plt.title(f'Confusion matrix ({prefix} Normalized) alpha={alpha:.2f}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{prefix}_normalized_alpha{alpha:.2f}.png'),
                dpi=600, bbox_inches='tight')
    plt.close()

    # ---------- Compute evaluation metrics ----------
    correct = C.diagonal(offset=0)
    per_row = C.sum(axis=1)
    per_col = C.sum(axis=0)
    Precision = [0] * 7
    Recall = [0] * 7
    F1 = [0] * 7
    for i in range(7):
        Precision[i] = correct[i] / per_col[i] if per_col[i] != 0 else 0
        Recall[i] = correct[i] / per_row[i] if per_row[i] != 0 else 0
        F1[i] = (2 * Precision[i] * Recall[i]) / (Precision[i] + Recall[i]) if (Precision[i] + Recall[i]) != 0 else 0
    macro_avg = [sum(Precision) / 7, sum(Recall) / 7, sum(F1) / 7]
    total = len(labels)
    weighted_avg = [
        sum(Precision[i] * per_row[i] for i in range(7)) / total,
        sum(Recall[i] * per_row[i] for i in range(7)) / total,
        sum(F1[i] * per_row[i] for i in range(7)) / total,
    ]

    # ---------- Save text report ----------
    with open(os.path.join(OUTPUT_DIR, f'test_results_alpha{alpha:.2f}.txt'), 'w') as f:
        f.write(f"Viterbi decoding results (alpha = {alpha:.2f})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Viterbi accuracy: {acc:.4f}\n")
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(C, separator=', '))
        f.write("\n\nSamples per true class: " + str(per_row))
        f.write("\nPrecision: " + str([f'{p:.4f}' for p in Precision]))
        f.write("\nRecall: " + str([f'{r:.4f}' for r in Recall]))
        f.write("\nF1 score: " + str([f'{f:.4f}' for f in F1]))
        f.write("\nMacro average: " + str([f'{m:.4f}' for m in macro_avg]))
        f.write("\nWeighted average: " + str([f'{w:.4f}' for w in weighted_avg]))
    print(f"Saved results for alpha={alpha:.2f}")


# ============================================================
# Main workflow
# ============================================================
def main():
    # ========== Device configuration ==========
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ========== Image transforms ==========
    data_transform = {
        'test': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
    }

    # ========== Load model ==========
    model = ResNet50().to(device)

    try:
        # Loading a full model on PyTorch 2.6+ requires weights_only=False; older versions may not accept this keyword
        try:
            model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        except TypeError:
            model = torch.load(MODEL_PATH, map_location=device)
        print(f"Successfully loaded full model: {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load full model: {e}")
        print("Attempting to load model weights...")
        state_dict = torch.load(PARAMS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights: {PARAMS_PATH}")

    model.eval()

    # Print model parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 50}")
    print("Model parameter counts")
    print(f"{'=' * 50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"{'=' * 50}\n")

    # ========== 5. Mode 1: independent image predictions ==========
    print("=" * 60)
    print("【Mode 1】Independent image prediction")
    print("=" * 60)

    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, 'test'),
        transform=data_transform['test']
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

    y_t = []
    y_p = []
    acc_count = 0
    count = 0

    with torch.no_grad():
        for batch, data in enumerate(test_loader, 1):
            X, Y = data
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)
            count += Y.size(0)
            acc_count += (pred == Y).sum().item()
            y_t.append(Y.cpu())
            y_p.append(pred.cpu())
            print(f'Batch {batch} - original label:{Y.cpu().numpy()}, predict label:{pred.cpu().numpy()}')

    test_acc = acc_count / count if count != 0 else 0
    print(f"\nIndependent image prediction - Total test samples: {count}")
    print(f"Independent image prediction - Test accuracy: {test_acc:.4f}")

    # Confusion matrix and metrics for independent predictions
    y_t = torch.cat(y_t).tolist()
    y_p = torch.cat(y_p).tolist()
    y_tt = [str(i) for i in y_t]
    y_pp = [str(i) for i in y_p]

    C = confusion_matrix(y_tt, y_pp)
    print("\nConfusion matrix (counts):")
    print(C)

    # Count-based confusion matrix plot
    plt.matshow(C, cmap=plt.cm.Greens)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.colorbar()
    plt.ylabel('True label', size=12)
    plt.xlabel('Predicted label', size=12)
    plt.xticks(np.arange(7), labels=['St', 'Sa', 'Sp', 'Sw', 'Fw', 'Fh', 'O'])
    plt.yticks(np.arange(7), labels=['St', 'Sa', 'Sp', 'Sw', 'Fw', 'Fh', 'O'])
    plt.title('Confusion matrix (Single)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_single.png'), dpi=600, bbox_inches='tight')
    plt.show()

    # Normalized confusion matrix
    C_norm = C.astype(np.float64)
    s = C_norm.sum(axis=1)
    for i in range(len(C_norm)):
        for j in range(len(C_norm)):
            C_norm[i][j] = np.round(((C_norm[i][j] / s[i]) * 100), 1) if s[i] != 0 else 0

    plt.matshow(C_norm, cmap=plt.cm.Greens)
    for i in range(len(C_norm)):
        for j in range(len(C_norm)):
            plt.annotate(C_norm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.colorbar()
    plt.ylabel('True label', size=12)
    plt.xlabel('Predicted label', size=12)
    plt.xticks(np.arange(7), labels=['St', 'Sa', 'Sp', 'Sw', 'Fw', 'Fh', 'O'])
    plt.yticks(np.arange(7), labels=['St', 'Sa', 'Sp', 'Sw', 'Fw', 'Fh', 'O'])
    plt.title('Confusion matrix (Single Normalized)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_single_normalized.png'),
                dpi=600, bbox_inches='tight')
    plt.show()

    # Performance metrics
    correct = C.diagonal(offset=0)
    per_row = C.sum(axis=1)
    per_col = C.sum(axis=0)
    Precision = [0] * 7
    Recall = [0] * 7
    F1 = [0] * 7
    macro_avg = [0] * 3
    weighted_avg = [0] * 3
    psum = rsum = fsum = 0

    for i in range(7):
        Precision[i] = correct[i] / per_col[i] if per_col[i] != 0 else 0
        Recall[i] = correct[i] / per_row[i] if per_row[i] != 0 else 0
        F1[i] = (2 * Precision[i] * Recall[i]) / (Precision[i] + Recall[i]) if (Precision[i] + Recall[i]) != 0 else 0
        psum += Precision[i] * per_row[i]
        rsum += Recall[i] * per_row[i]
        fsum += F1[i] * per_row[i]

    macro_avg[0] = sum(Precision) / 7
    macro_avg[1] = sum(Recall) / 7
    macro_avg[2] = sum(F1) / 7
    weighted_avg[0] = psum / count if count != 0 else 0
    weighted_avg[1] = rsum / count if count != 0 else 0
    weighted_avg[2] = fsum / count if count != 0 else 0

    print('\nIndependent image prediction - Samples per true class:', per_row)
    print('Independent image prediction - Precision:', [f'{p:.4f}' for p in Precision])
    print('Independent image prediction - Recall:', [f'{r:.4f}' for r in Recall])
    print('Independent image prediction - F1 score:', [f'{f:.4f}' for f in F1])
    print('Independent image prediction - Macro average:', [f'{m:.4f}' for m in macro_avg])
    print('Independent image prediction - Weighted average:', [f'{w:.4f}' for w in weighted_avg])

    # ========== 6. Mode 2: Viterbi sequence decoding (fixed alpha) ==========
    print("\n" + "=" * 60)
    print("【Mode 2】Viterbi sequence decoding (grouped by well)")
    print("=" * 60)

    # Parse well IDs and depths from ImageFolder samples and group by well
    well_groups = defaultdict(list)
    for idx, (path, label) in enumerate(test_dataset.samples):
        fname = os.path.basename(path)
        name_no_ext = fname.rsplit('.', 1)[0]
        parts = name_no_ext.split('_')
        if len(parts) < 3:
            continue
        well_id = parts[0]
        depth = float(parts[1])
        well_groups[well_id].append((depth, idx, label))

    # Convert the learned transition matrix to log probabilities
    trans_log = F.log_softmax(model.trans_matrix.detach(), dim=1).cpu()

    print("\n" + "-" * 60)
    print(f"【Viterbi decoding with fixed alpha = {SELECTED_ALPHA:.2f}】")
    print("-" * 60)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for well_id, group in well_groups.items():
            group.sort(key=lambda x: x[0])
            indices = [g[1] for g in group]
            labels = [g[2] for g in group]
            logits_list = []
            for idx in indices:
                img, _ = test_dataset[idx]
                img = img.unsqueeze(0).to(device)
                logit = model(img).squeeze(0).cpu()
                logits_list.append(logit)
            pred_seq = viterbi_decode(logits_list, trans_log, alpha=SELECTED_ALPHA)
            all_preds.extend(pred_seq)
            all_labels.extend(labels)

    viterbi_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"alpha={SELECTED_ALPHA:.2f}, Viterbi Acc={viterbi_acc:.4f}")
    print("-" * 60)

    # Generate confusion matrices and metrics for the selected alpha
    generate_and_save_metrics(all_preds, all_labels, SELECTED_ALPHA, viterbi_acc, prefix='Viterbi')

    # ========== 7. Compare the two inference modes ==========
    print("\n" + "=" * 60)
    print("【Test results summary】")
    print("=" * 60)
    print(f"Independent image prediction accuracy: {test_acc:.4f}")
    print(f"Viterbi sequence decoding accuracy (alpha={SELECTED_ALPHA:.2f}): {viterbi_acc:.4f}")
    print(f"Sequence-constraint gain: {(viterbi_acc - test_acc) * 100:.2f} percentage points")
    print("=" * 60)

    # Save the summary to a file
    with open(os.path.join(OUTPUT_DIR, 'test_results_summary.txt'), 'w') as f:
        f.write("Test results summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Independent image prediction accuracy: {test_acc:.4f}\n")
        f.write(f"Viterbi sequence decoding (alpha={SELECTED_ALPHA:.2f}) accuracy: {viterbi_acc:.4f}\n")
        f.write(f"Sequence-constraint gain: {(viterbi_acc - test_acc) * 100:.2f} percentage points\n")
    print("Summary saved to test_results_summary.txt")
    print("All figures were saved as PNG files.")


if __name__ == '__main__':
    main()