import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Dataset ----------

class SkeletonSequenceDataset(Dataset):
    def __init__(self, csv_files, seq_len=120, step=15, label_map=None):
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        self.data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        self.seq_len = seq_len
        self.step = step
        self.label_map = label_map or {
            'Sitting quietly': 0, 'Using phone': 1, 'Walking': 2, 'Eating snacks': 3,
            'Head banging': 4, 'Throwing things': 5, 'Attacking': 6, 'Biting': 7
        }

        self.sequences = []
        self.labels = []
        self.prepare_sequences()

    def prepare_sequences(self):
        num_frames = len(self.data)
        joint_cols = self.data.columns[1:-1]

        for start in range(0, num_frames - self.seq_len + 1, self.step):
            end = start + self.seq_len
            seq_df = self.data.iloc[start:end]

            if seq_df['Action Label'].isnull().all():
                continue

            mode_label = seq_df['Action Label'].mode()
            if mode_label.empty:
                continue

            label_str = mode_label.iloc[0]
            if label_str not in self.label_map:
                continue

            coords = seq_df[joint_cols].values.reshape(self.seq_len, -1, 2)  # [T, V, 2]

            # Normalize đoạn này: chuẩn hóa toàn bộ đoạn theo mean/std
            coords_mean = coords.mean(axis=(0, 1), keepdims=True)  # [1, 1, 2]
            coords_std = coords.std(axis=(0, 1), keepdims=True) + 1e-6
            coords = (coords - coords_mean) / coords_std

            label = self.label_map[label_str]

            self.sequences.append(coords)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # [T, V, 2]

        # ----- Tính velocity (dx, dy) -----
        dxdy = seq[1:] - seq[:-1]             # [T-1, V, 2]
        zero_padding = np.zeros_like(seq[0:1])  # [1, V, 2] để giữ T khớp nhau
        dxdy = np.concatenate([dxdy, zero_padding], axis=0)  # [T, V, 2]

        # ----- Ghép x, y và dx, dy → thành [x, y, dx, dy] -----
        seq_with_vel = np.concatenate([seq, dxdy], axis=-1)  # [T, V, 4]

        # ----- Trả về tensor -----
        seq_tensor = torch.tensor(seq_with_vel, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return seq_tensor, label_tensor

    
# ---------- GraphSAGE Layer ----------
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, out_feats, adj, agg_func='mean'):
        super().__init__()
        self.adj = adj  # [V, V]
        self.fc = nn.Linear(8, 64)
        self.agg_func = agg_func

    def forward(self, x):  # x: [B, T, V, C]
        B, T, V, C = x.shape
        x = x.view(B * T, V, C)

        # Lấy neighbor feature: [B*T, V, C]
        A = self.adj.to(x.device)  # [V, V]
        neighbor_feat = torch.matmul(A, x)  # tổng (hoặc trung bình) neighbor

        if self.agg_func == 'mean':
            degree = A.sum(dim=-1, keepdim=True).clamp(min=1)
            neighbor_feat = neighbor_feat / degree  # trung bình neighbor

        combined = torch.cat([x, neighbor_feat], dim=-1)  # [B*T, V, 2C]
        out = self.fc(combined)  # [B*T, V, out_feats]
        return out.view(B, T, V, -1)

# ---------- GCN Layer ----------

class SpatialGCN(nn.Module):
    def __init__(self, in_features, out_features, A):
        super().__init__()
        self.A = A
        self.gc = nn.Linear(in_features, out_features)

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.view(B * T, N, C)
        x = torch.matmul(self.A.to(x.device), x)
        x = self.gc(x)
        x = x.view(B, T, N, -1)
        return x


# ---------- Transformer ----------

class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, max_len=500):
        super().__init__()
        self.pos_embed = nn.Parameter(self.get_positional_encoding(max_len, embed_dim), requires_grad=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def get_positional_encoding(self, max_len, d_model):
        # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        return pe  # Không học

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        x = x + self.pos_embed[:, :T, :].to(x.device)
        return self.encoder(x)





class PartBasedCNN(nn.Module):
    def __init__(self, joint_groups, in_channels=4, out_channels=8):  # 👈 Sửa chỗ này
        super().__init__()
        self.extractors = nn.ModuleDict({
            part: nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            for part in joint_groups
        })
        self.joint_groups = joint_groups
        self.total_out = len(joint_groups) * out_channels


    def forward(self, x):  # [B, T, V, 2]
        B, T, V, C = x.shape
        part_feats = []

        for part, joints in self.joint_groups.items():
            sub = x[:, :, joints, :]  # [B, T, len_joints, 2]
            sub = sub.reshape(B * T, len(joints), C).permute(0, 2, 1)  # [B*T, 2, len_joints]
            feat = self.extractors[part](sub)  # [B*T, out_channels, 1]
            feat = feat.squeeze(-1).view(B, T, -1)  # [B, T, out_channels]
            part_feats.append(feat)

        return torch.cat(part_feats, dim=-1)  # [B, T, total_out]


# ---------- Temporal Attention (Học được) ----------
# 1. Thêm temporal attention (học được):
class TemporalAttention(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.seq_len = seq_len

    def forward(self, x):  # x: [B, T, D]
        B, T, D = x.shape
        attn_scores = self.attn(x)  # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T, 1]
        weighted = x * attn_weights  # [B, T, D]
        return weighted.sum(dim=1)  # [B, D]
    
# ---------- Mouth Region CNN (Học được) ----------
class MouthRegionCNN(nn.Module):
    def __init__(self, in_channels=4, out_dim=128):  # 👈 sửa tại đây (64 → 128)
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 1), padding=(1, 0))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_dim)



    def forward(self, x):  # (B, C=4, T, J=3)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # (B, out_dim)


def get_ear_wrist_distance(x):  # x: [B, T, V, C]
    left_ear = x[:, :, 1, :]     # [B, T, 2]
    right_ear = x[:, :, 2, :]
    left_wrist = x[:, :, 5, :]
    right_wrist = x[:, :, 8, :]

    left_dist = torch.norm(left_ear - left_wrist, dim=-1, keepdim=True)  # [B, T, 1]
    right_dist = torch.norm(right_ear - right_wrist, dim=-1, keepdim=True)

    return torch.cat([left_dist, right_dist], dim=-1)  # [B, T, 2]
def get_extra_part_distance(x):  # x: [B, T, V, C]
    left_knee = x[:, :, 13, :]     # bạn cần xác định đúng ID của các khớp
    left_ankle = x[:, :, 15, :]
    right_knee = x[:, :, 14, :]
    right_ankle = x[:, :, 16, :]

    left_elbow = x[:, :, 3, :]
    left_wrist = x[:, :, 5, :]
    right_elbow = x[:, :, 6, :]
    right_wrist = x[:, :, 8, :]

    head = x[:, :, 0, :]
    left_shoulder = x[:, :, 4, :]
    right_shoulder = x[:, :, 7, :]
    shoulder_center = (left_shoulder + right_shoulder) / 2

    # distance per pair
    knee_ankle = torch.norm(left_knee - left_ankle, dim=-1, keepdim=True)
    elbow_wrist = torch.norm(left_elbow - left_wrist, dim=-1, keepdim=True)
    head_shoulder = torch.norm(head - shoulder_center, dim=-1, keepdim=True)

    return torch.cat([knee_ankle, elbow_wrist, head_shoulder], dim=-1)  # [B, T, 3]

# ---------- Model ----------

class BehaviorClassifier(nn.Module):
    def __init__(self, num_joints, input_dim=4, gcn_out=64, transformer_embed=128, num_classes=8, seq_len=90):
        super().__init__()

        self.extra_part_linear = nn.Linear(3, 64)  # [B, T, 3] → [B, T, 64]

        self.adj = self.build_adjacency(num_joints)
        self.gcn = GraphSAGE(input_dim, gcn_out, self.adj)

        self.joint_groups = {
            'nose': [0],
            'head': [1, 2],
            'left_wrist': [5],
            'right_wrist': [8],
            'left_arm': [3, 4],
            'right_arm': [6, 7],
            'left_leg': [9, 10, 11],
            'right_leg': [12, 13, 14],
            'torso': [15, 16]
        }

        self.part_cnn = PartBasedCNN(self.joint_groups, in_channels=input_dim, out_channels=8)
        self.mouth_cnn = MouthRegionCNN(in_channels=input_dim, out_dim=128)  # 👈 sửa ở đây (64 → 128)


        self.ear_wrist_linear = nn.Linear(2, 16)

        total_feat = gcn_out + self.part_cnn.total_out + 128 + 16 + 64  # 👈 sửa 64 → 128




        self.fc_embed = nn.Linear(total_feat, transformer_embed)
        self.transformer = TemporalTransformer(transformer_embed, num_heads=4, depth=2)
        self.attn_pool = TemporalAttention(seq_len=seq_len, embed_dim=transformer_embed)

        self.classifier = nn.Sequential(
            nn.Linear(transformer_embed, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def build_adjacency(self, N):
        A = torch.eye(N)
        for i in range(N - 1):
            A[i, i + 1] = A[i + 1, i] = 1
        return A

    def forward(self, x):
        B, T, V, C = x.shape
        weight = torch.ones((B, T, V, 1), device=x.device)
        weight[:, :, 0, :] *= 3.0  # nose
        weight[:, :, 5, :] *= 5.0  # left_wrist
        weight[:, :, 8, :] *= 5.0  # right_wrist
        weight[:, :, [1, 2], :] *= 3.5  # left ear, right ear
        weight[:, :, [3, 4, 6, 7], :] *= 1.8
        weight[:, :, [9, 10, 11, 12, 13, 14], :] *= 1.2
        x = x * weight

        ear_wrist_feat = self.ear_wrist_linear(get_ear_wrist_distance(x))  # [B, T, 16]
        gcn_feat = self.gcn(x).mean(dim=2)
        part_feat = self.part_cnn(x)
        # Extract 3 joints: nose + head (joints 0,1,2)
        mouth_joints = x[:, :, [0, 1, 2], :]  # [B, T, 3, 2]
        mouth_input = mouth_joints.permute(0, 3, 1, 2)  # [B, C=2, T, J=3]
        mouth_feat = self.mouth_cnn(mouth_input)  # [B, out_dim=64]
        mouth_feat = mouth_feat.unsqueeze(1).expand(-1, T, -1)  # [B, T, 64]


        extra_part_feat = self.extra_part_linear(get_extra_part_distance(x))  # [B, T, 64]

        fused = torch.cat([
            gcn_feat,
            part_feat,
            mouth_feat,
            ear_wrist_feat,
            extra_part_feat  # thêm dòng này
            ],dim=-1)

        fused = self.fc_embed(fused)
        fused = self.transformer(fused)
        pooled = self.attn_pool(fused)
        return self.classifier(pooled)

# ✅ Đã tích hợp Positional Encoding (sin-cos) + Focal Loss + Temporal Attention + Weighted joints
# Gợi ý tiếp theo:
# - Nếu vẫn nhầm nhiều về 'biting', 'attacking', hãy thử visualize attention map hoặc tách riêng CNN cho tay/head



# ---------- Focal Loss ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0, num_classes=8):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)  # [B, C]

        if self.label_smoothing > 0:
            # One-hot với smoothing
            smooth_target = torch.zeros_like(log_probs).scatter(1, targets.unsqueeze(1), 1)
            smooth_target = smooth_target * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        else:
            # Chuẩn one-hot
            smooth_target = torch.zeros_like(log_probs).scatter(1, targets.unsqueeze(1), 1)

        probs = torch.exp(log_probs)
        pt = (probs * smooth_target).sum(dim=-1)  # [B]
        CE_loss = -(log_probs * smooth_target).sum(dim=-1)

        focal_loss = (1 - pt) ** self.gamma * CE_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]

            focal_loss *= alpha_t

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def plot_confusion_matrix_inline(model, loader, device, label_map, save_path):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Best Model Confusion Matrix (Val Set)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved at {save_path}")

# ---------- Training ----------
from collections import Counter

def train(model, train_loader, val_loader, device, num_epochs=30, label_map=None, save_path="best_model.pth"):
    label_counts = Counter(train_set.labels)
    total = sum(label_counts.values())
    weights = [total / label_counts[i] for i in range(len(label_counts))]
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    alpha = torch.tensor([2.5, 2.5, 0.5, 1.0, 2.5, 2.0, 3.0, 4.0])

    criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.1, num_classes=8)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total if total > 0 else 0.0

        print(f"\n[Epoch {epoch+1}]")
        print(f"Training Loss: {total_loss / len(train_loader):.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")

        f1 = evaluate(model, val_loader, device, label_map)

        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved with F1: {best_f1:.4f}")
            plot_confusion_matrix_inline(model, val_loader, device, label_map, save_path="best_confusion_val.png")

        else:
            print(f"No improvement (Best F1: {best_f1:.4f})")


# ---------- Evaluation ----------

def evaluate(model, loader, device, label_map=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    print(f"\nValidation/Test Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    if label_map:
        inv_label_map = {v: k for k, v in label_map.items()}
        labels = sorted(inv_label_map.keys())
        target_names = [inv_label_map[i] for i in labels]

        print("\nPer-Class Report:")
        print(classification_report(
            all_labels, all_preds,
            labels=labels,
            target_names=target_names,
            zero_division=0
        ))

    return f1


# ---------- Confusion Matrix ----------
def plot_confusion_matrix(model_class, model_path, loader, device, label_map, save_path=None, title="Confusion Matrix"):
    # Khởi tạo model và load best weights
    num_joints = 17  # hoặc truyền vào nếu cần
    model = model_class(num_joints)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Dự đoán
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Tạo confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    # Vẽ
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Confusion matrix saved to {save_path}")
    else:
        plt.show()


# ---------- Run Everything ----------

if __name__ == "__main__":
    # ✅ Đường dẫn theo yêu cầu
    train_csvs = [
        "./data/keypointlabel/keypoints_with_labels_1.csv",
        "./data/keypointlabel/keypoints_with_labels_2.csv",
        "./data/keypointlabel/keypoints_with_labels_3.csv"
    ]
    test_csvs = ["./data/keypointlabel/keypoints_with_labels_5.csv"]


    # ✅ Tạo Dataset
    train_set = SkeletonSequenceDataset(train_csvs, seq_len=90, step=15)
    val_set = SkeletonSequenceDataset(train_csvs, seq_len=90, step=15)  # Dùng lại train_csvs làm val nếu không có val riêng
    test_set = SkeletonSequenceDataset(test_csvs, seq_len=90, step=15)

    # ✅ DataLoader
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    # ✅ Model
    num_joints = 17
    model = BehaviorClassifier(num_joints=17, input_dim=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # --- Đóng băng backbone nếu muốn ---
    # for name, param in model.named_parameters():
    #     if "cnn" in name or "gcn" in name or "sage" in name:
    #         param.requires_grad = False

    model.to(device)

    # ✅ Optimizer chỉ cho phần còn lại
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5,  # như bạn yêu cầu
        weight_decay=1e-5
    )

    # ✅ Train
    train(model, train_loader, val_loader, device, num_epochs=30, label_map=train_set.label_map)

    # ✅ Evaluate on test set
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)

    print("\n🎯 Final Evaluation on Test Set:")
    evaluate(model, test_loader, device, train_set.label_map)
    plot_confusion_matrix(
        model_class=BehaviorClassifier,
        model_path="best_model.pth",
        loader=test_loader,
        device=device,
        label_map=train_set.label_map,
        save_path="confusion_matrix.png",
        title="Best Model - Confusion Matrix (Test Set)"
    )
