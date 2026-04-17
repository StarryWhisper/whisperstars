import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(1, channel // reduction)
        self.fc1 = nn.Linear(channel, hidden, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        squeeze = x.view(b, c, -1).mean(dim=2)
        excitation = self.fc2(self.relu(self.fc1(squeeze)))
        scale = self.sigmoid(excitation).view(b, c, 1, 1)
        return x * scale


class ResNeXtWithAttention(nn.Module):
    """Best model used by train_best.py."""

    def __init__(self, num_classes=100):
        super().__init__()
        try:
            backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        except AttributeError:
            backbone = models.resnext50_32x4d(pretrained=True)

        # CIFAR-100 stem adaptation (32x32)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        num_ftrs = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.se_block = SEBlock(num_ftrs, reduction=16)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.se_block(features)
        x = features.view(features.size(0), -1)
        return self.classification_head(x)


def build_test_loader(batch_size, num_workers, data_root):
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_dataset, test_loader


def run_eval(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="测试评估"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100.0
    return accuracy, all_labels, all_preds


def save_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best CIFAR-100 model checkpoint")
    parser.add_argument("--model-path", default="best_cifar_model.pth", help="Checkpoint path")
    parser.add_argument("--data-root", default="./data", help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--output-dir", default=".", help="Directory to save reports")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    test_dataset, test_loader = build_test_loader(args.batch_size, args.num_workers, args.data_root)

    model = ResNeXtWithAttention(num_classes=100).to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    accuracy, labels, preds = run_eval(model, test_loader, device)

    report = classification_report(labels, preds, target_names=test_dataset.classes, output_dict=True)
    report_text = classification_report(labels, preds, target_names=test_dataset.classes)
    cm = confusion_matrix(labels, preds)

    print(f"测试集准确率: {accuracy:.2f}%")
    print("\n分类报告:")
    print(report_text)

    metrics_path = output_dir / "best_model_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy_percent": round(accuracy, 4),
                "num_samples": len(labels),
                "classification_report": report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    cm_path = output_dir / "best_model_confusion_matrix.png"
    save_confusion_matrix(cm, test_dataset.classes, cm_path)

    print(f"\n已保存指标: {metrics_path}")
    print(f"已保存混淆矩阵: {cm_path}")


if __name__ == "__main__":
    main()
