# CIFAR-100 图像分类实验说明

本项目使用 PyTorch 在 CIFAR-100 数据集上进行图像分类，包含训练、验证、测试、混淆矩阵与错误样本分析等完整流程。

---

## 1. 项目结构

```text
Task1_Image_Classification/
├── train.py
├── train_best.py
├── performance_summary.md
├── promote.txt
├── data/
│   └── cifar-100-python/
├── confusion_matrix.png
├── training_curves.png
├── error_analysis.png
└── error_correct_samples.png
```

---

## 2. 环境要求

- OS: Linux（当前环境）
- Python: 建议 3.9+
- 深度学习框架: PyTorch + torchvision
- 其他依赖: numpy, matplotlib, scikit-learn, seaborn, tqdm, pandas

可按如下方式创建并激活虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
```

安装依赖：

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn seaborn tqdm
```

## 3. 数据集说明

代码使用 `torchvision.datasets.CIFAR100` 自动下载数据：

- 训练集（50000）用于训练/验证划分
- 测试集（10000）用于最终评估

下载目录默认在：

- `./data`

## 4. 快速开始

### 4.1 运行推荐版本

```bash
source venv/bin/activate
python3 train_best.py
```

训练过程中会输出每个 epoch 的：

- 训练损失、训练准确率
- 验证损失、验证准确率
- 早停（Early Stopping）信息

并保存最佳权重：

- `best_cifar_model.pth`

## 5. 输出结果文件

训练/评估后通常会生成：

- `best_cifar_model.pth`：验证集最佳模型权重
- `training_curves.png`：训练/验证损失与准确率曲线
- `confusion_matrix.png`：测试集混淆矩阵
- `error_analysis.png`：错误分类样本可视化
- `error_correct_samples.png`：正确与错误样本对比
