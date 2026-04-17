# 基础科学计算与数据处理
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch 核心
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# PyTorch 数据加载与处理
from torchvision import datasets, transforms, models

# 评估指标
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 进度条
from tqdm import tqdm

# Mixup 和 CutMix 数据增强
from numpy.random import beta as beta_dist
import numpy as np

def rand_bbox(size, lam):
    """生成随机边界框用于 CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(images, labels, alpha=1.0):
    """CutMix 数据增强"""
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    mixed_images = images.clone()
    mixed_labels = labels.clone()
    
    lam = beta_dist(alpha, alpha)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
    
    # 调整 lambda 以考虑实际裁剪区域
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    
    return mixed_images, (labels, labels[index], lam)

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 1. 定义数据预处理/增强变换（增强版，添加更多变换）
train_transform = transforms.Compose([
    # 几何变换 (PIL)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomCrop(32, padding=4),
    
    transforms.ToTensor(),
    
    # 张量变换
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    
    # 颜色增强
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.1
    ),
    
    # 高级增强
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    transforms.RandomErasing(p=0.12, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# 2. 下载并加载 CIFAR-100 数据集
print("正在下载/加载 CIFAR-100 数据集...")
# 先下载一次无 transform 的训练集合，用于稳定划分索引
train_val_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)

# 为训练/验证分别构建数据集，避免验证阶段使用随机增强
train_dataset_full = datasets.CIFAR100(root='./data', train=True, download=False, transform=train_transform)
val_dataset_full = datasets.CIFAR100(root='./data', train=True, download=False, transform=test_transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

# 3. 划分训练集和验证集 (例如 80% 训练, 20% 验证)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
index_generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(train_val_dataset), generator=index_generator).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)

# 4. 创建数据加载器 (DataLoader) - 优化批大小
batch_size = 96  # 稍微减小批大小以获得更好的梯度估计
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 数据信息
print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")
print(f"类别名称: {train_val_dataset.classes}")

# 定义 Squeeze-and-Excitation (SE) 注意力模块
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # 全局平均池化 + FC层用于通道重新标定
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, C, H, W)
        batch_size, num_channels, _, _ = x.size()
        
        # Squeeze: 全局平均池化
        squeeze = x.view(batch_size, num_channels, -1).mean(dim=2)  # (B, C)
        
        # Excitation: FC层学习通道重要性
        excitation = self.fc1(squeeze)  # (B, C//16)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)  # (B, C)
        excitation = self.sigmoid(excitation)  # (B, C)
        
        # 重新标定: 通道加权
        scale = excitation.view(batch_size, num_channels, 1, 1)  # (B, C, 1, 1)
        return x * scale

# 方案 B: 使用 ResNeXt-50 (经过优化的配置，添加注意力层)
class ResNeXtWithAttention(nn.Module):
    """ResNeXt-50 with SE Attention modules"""
    def __init__(self, num_classes=100):
        super(ResNeXtWithAttention, self).__init__()
        # 加载预训练的 ResNeXt-50（兼容新旧 torchvision API）
        try:
            self.backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        except AttributeError:
            self.backbone = models.resnext50_32x4d(pretrained=True)

        # CIFAR-100 为 32x32，小尺寸输入下建议减弱 stem 下采样
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        # 获取 ResNeXt 的特征维度
        num_ftrs = self.backbone.fc.in_features
        
        # 移除原始的 FC 层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 添加 SE 注意力模块
        self.se_block = SEBlock(num_ftrs, reduction=16)
        
        # 添加新的分类头（带 dropout）
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # 通过 ResNeXt 主干
        features = self.backbone(x)  # (B, 2048, 1, 1)
        
        # 应用 SE 注意力
        attended_features = self.se_block(features)  # (B, 2048, 1, 1)
        
        # 展平
        x = attended_features.view(attended_features.size(0), -1)  # (B, 2048)
        
        # 分类头
        x = self.classification_head(x)
        return x

def get_resnext_model(num_classes=100):
    """获取ResNeXt-50模型，添加SE注意力和强正则化"""
    model = ResNeXtWithAttention(num_classes=num_classes)
    return model

# 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = get_resnext_model(num_classes=100).to(device)

# 定义训练参数
num_epochs = 50  # 先定义，用于学习率调度器

# 2. 定义损失函数和优化器 (优化配置)
# 使用 Label Smoothing 来防止过拟合
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 更小的学习率，配合更长的预热期
base_lr = 0.0002
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

# 学习率预热 + 余弦退火调度
def lr_lambda(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        # 预热阶段：线性增加到 base_lr
        return (epoch + 1) / warmup_epochs
    else:
        # 余弦退火阶段
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def mixup(images, labels, alpha=0.2):
    """Mixup 数据增强"""
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    mixed_images = images.clone()
    mixed_labels = labels.clone()
    
    lam = beta_dist(alpha, alpha)
    mixed_images = lam * images + (1 - lam) * images[index, :]
    
    return mixed_images, (labels, labels[index], lam)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 动态调整增强强度（后期训练使用更强的增强）
    mixup_alpha = 0.2 + 0.3 * (epoch / num_epochs)  # 随着训练进行增加 mixup 强度
    cutmix_alpha = 1.0 + 0.5 * (epoch / num_epochs)  # 增加 cutmix 强度

    for images, labels in tqdm(loader, desc='训练'):
        images, labels = images.to(device), labels.to(device)
        
        # 动态增强策略
        rand_num = torch.rand(1).item()
        if rand_num < 0.25:  # 25% CutMix
            images, mixed_labels = cutmix(images, labels, alpha=cutmix_alpha)
            y_a, y_b, lam = mixed_labels
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        elif rand_num < 0.5:  # 25% Mixup
            images, mixed_labels = mixup(images, labels, alpha=mixup_alpha)
            y_a, y_b, lam = mixed_labels
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:  # 50% 标准训练
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='验证'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# 开始训练（优化配置，增加训练轮数）
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_acc = 0.0
patience = 10  # 增加 patience
no_improve = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
    scheduler.step() # 调整学习率

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
    print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')

    # 保存验证集上性能最好的模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_cifar_model.pth')
        print(f'模型已保存 (当前最佳验证准确率: {val_acc:.2f}%)')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# 1. 加载最佳模型并在独立测试集上评估
print("\n=== 在测试集上进行最终评估 ===")
model.load_state_dict(torch.load('best_cifar_model.pth'))
test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
print(f'测试集准确率: {test_acc:.2f}%')

# 2. 生成混淆矩阵和分类报告
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.show()

print("\n分类报告:")
print(classification_report(test_labels, test_preds, target_names=train_val_dataset.classes))

# 3. 错误样本分析
errors = []
for i, (pred, true) in enumerate(zip(test_preds, test_labels)):
    if pred != true:
        errors.append((i, pred, true))

print(f"\n错误样本数量: {len(errors)} / {len(test_labels)}")

# 显示正确和错误样本的对比
correct_samples = []
for i, (pred, true) in enumerate(zip(test_preds, test_labels)):
    if pred == true:
        correct_samples.append(i)

print(f"\n正确样本数量: {len(correct_samples)} / {len(test_labels)}")

if len(errors) > 0 and len(correct_samples) > 0:
    fig, axes = plt.subplots(4, 5, figsize=(16, 10))
    axes = axes.flatten()
    
    # 显示前5个正确的样本
    for idx in range(min(5, len(correct_samples))):
        i = correct_samples[idx]
        img, _ = test_dataset[i]
        img = img.permute(1, 2, 0).numpy()
        img = (img * np.array([0.2675, 0.2565, 0.2761]) + np.array([0.5071, 0.4867, 0.4408]))  # 反归一化
        img = np.clip(img, 0, 1)
        axes[idx].imshow(img)
        true_label = test_labels[i]
        axes[idx].set_title(f'✓ {train_val_dataset.classes[true_label]}', color='green', fontweight='bold')
        axes[idx].axis('off')
    
    # 显示前15个错误的样本
    for idx, (i, pred, true) in enumerate(errors[:15]):
        img, _ = test_dataset[i]
        img = img.permute(1, 2, 0).numpy()
        img = (img * np.array([0.2675, 0.2565, 0.2761]) + np.array([0.5071, 0.4867, 0.4408]))  # 反归一化
        img = np.clip(img, 0, 1)
        axes[5 + idx].imshow(img)
        axes[5 + idx].set_title(f'✗ True: {train_val_dataset.classes[true]}\nPred: {train_val_dataset.classes[pred]}', 
                                color='red', fontsize=8)
        axes[5 + idx].axis('off')
    
    plt.suptitle('正确 (绿色) 和错误 (红色) 样本对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('error_correct_samples.png', dpi=100)
    plt.show()

# 2. 绘制训练曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Val Loss')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(train_accs, label='Train Acc')
axes[1].plot(val_accs, label='Val Acc')
axes[1].set_title('Accuracy over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
plt.tight_layout()
plt.savefig('training_curves.png') # 保存图片用于报告
plt.show()
