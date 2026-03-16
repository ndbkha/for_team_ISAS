import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os

from run import SkeletonSequenceDataset, BehaviorClassifier  # Import class từ file chính

def evaluate_and_plot(model, loader, device, label_map, save_path=None):
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

    # ---------- Metrics ----------
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    print("\n🎯 Test Set Evaluation:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # ---------- Classification Report ----------
    if label_map:
        inv_label_map = {v: k for k, v in label_map.items()}
        labels = sorted(inv_label_map.keys())
        target_names = [inv_label_map[i] for i in labels]

        print("\n📊 Per-Class Report:")
        print(classification_report(
            all_labels, all_preds,
            labels=labels,
            target_names=target_names,
            zero_division=0
        ))

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()

if __name__ == "__main__":
    # ----- Load test set -----
    csv_test = ["./data/keypointlabel/keypoints_with_labels_5.csv"]
    test_set = SkeletonSequenceDataset(csv_test)
    test_loader = DataLoader(test_set, batch_size=64)

    # ----- Load model -----
    num_joints = 17
    model = BehaviorClassifier(num_joints)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))  # dùng GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ----- Evaluate -----
    evaluate_and_plot(model, test_loader, device, test_set.label_map, save_path="results/conf_matrix_test.png")
