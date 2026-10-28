import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")          # 无图形界面环境下也能画图（服务器/后台运行更稳）
import matplotlib.pyplot as plt

# ══════════════════ 1. 所有可调参数都在这里 ══════════════════
PROJECT = r"D:\pytorch_project\fashion_mnist_project"
DATA_DIR = os.path.join(PROJECT, "fashion_mnist_data", "fashion")
SAVE_PATH = os.path.join(PROJECT, "outputs", "simple_cnn.pth")          # 最后一轮
BEST_PATH = os.path.join(PROJECT, "outputs", "simple_cnn_best.pth")     # 最优
CURVE_PATH = os.path.join(PROJECT, "outputs", "accuracy_curve.png")     # 曲线图

BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001

MEAN, STD = 0.2860, 0.3530

CLASSES = ["T恤/上衣", "裤子", "套头衫", "连衣裙", "外套",
           "凉鞋", "衬衫", "运动鞋", "包", "短靴"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════ 2. 读数据 ══════════════════
def data_file(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.isdir(path):
        path = os.path.join(path, os.listdir(path)[0])
    return path


def read_images(name):
    with open(data_file(name), "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, 28, 28)


def read_labels(name):
    with open(data_file(name), "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8, offset=8)


def to_tensor(images):
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    return (x - MEAN) / STD


# ══════════════════ 3. 模型 4层卷积2层全连接 ══════════════════
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ══════════════════ 4. 算准确率 ══════════════════
def accuracy(model, x, y):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), 1000):
            out = model(x[i:i + 1000].to(DEVICE))
            correct += (out.argmax(1).cpu() == y[i:i + 1000]).sum().item()
    return correct / len(x)


# ══════════════════ 4.5 画准确率曲线 ══════════════════
def plot_curve(train_accs, val_accs, save_path):
    """横轴是 epoch，两条线分别是训练集、验证集准确率。"""
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, marker="o", label="Train Acc")
    plt.plot(epochs, val_accs,   marker="s", label="Val Acc")

    # 标记最优验证准确率那一个点
    best_ep = int(np.argmax(val_accs)) + 1
    plt.scatter([best_ep], [val_accs[best_ep - 1]], color="red", zorder=5,
                label=f"Best Val {val_accs[best_ep - 1]:.4f} (epoch {best_ep})")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train / Val Accuracy")
    plt.xticks(list(epochs))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("曲线已保存到:", save_path)


# ══════════════════ 5. 主流程 ══════════════════
def main():
    print("设备:", DEVICE)

    # ---- 读数据 ----
    x_train = to_tensor(read_images("train-images-idx3-ubyte"))
    y_train = torch.tensor(read_labels("train-labels-idx1-ubyte"), dtype=torch.long)
    x_test = to_tensor(read_images("t10k-images-idx3-ubyte"))
    y_test = torch.tensor(read_labels("t10k-labels-idx1-ubyte"), dtype=torch.long)

    x_val, y_val = x_train[54000:], y_train[54000:]
    x_train, y_train = x_train[:54000], y_train[:54000]
    print(f"训练 {len(x_train)} 张，验证 {len(x_val)} 张，测试 {len(x_test)} 张")

    # ---- 模型、损失函数、优化器 ----
    model = CNN().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loader = DataLoader(TensorDataset(x_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True)

    # ---- 记录曲线用的容器 ----
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ---- 训练循环 ----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            out = model(xb)
            loss = loss_fn(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            seen += len(xb)

        train_acc = correct / seen
        val_acc = accuracy(model, x_val, y_val)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # —— 验证准确率刷新就存一份最优权重 ——
        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_PATH)
            flag = "  ← 保存最优"

        print(f"第 {epoch:2d} 轮  训练 loss {total_loss / seen:.4f}"
              f"  训练准确率 {train_acc:.4f}"
              f"  验证准确率 {val_acc:.4f}{flag}")

    # ---- 画曲线 ----
    plot_curve(train_accs, val_accs, CURVE_PATH)

    # ---- 用最优权重在测试集上评估一次（更有参考价值） ----
    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
    test_acc = accuracy(model, x_test, y_test)
    print(f"最优权重来自第 {best_epoch} 轮，验证准确率 {best_val_acc:.4f}")
    print("用最优权重在测试集上的准确率:", round(test_acc, 4))

    # ---- 顺便把最后一轮的权重也存一份 ----
    torch.save(model.state_dict(), SAVE_PATH)
    print("最优权重:", BEST_PATH)
    print("最后一轮权重:", SAVE_PATH)


if __name__ == "__main__":
    main()