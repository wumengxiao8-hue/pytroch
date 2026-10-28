import sys

import numpy as np
import torch
from PIL import Image

# 模型结构必须和训练时一模一样，所以直接复用 train.py 里的 CNN
from train import (CLASSES, MEAN, SAVE_PATH, STD, CNN, read_images, read_labels)

# ══════════════════ 1. 加载权重 ══════════════════
model = CNN()
# weights_only=True：只读张量，不执行权重文件里的任何代码，更安全（也是新版本的默认值）
model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu", weights_only=True))
model.eval()                      # 关掉训练专用的行为，进入推理模式
print("已加载权重:", SAVE_PATH)

# ══════════════════ 2. 准备一张图 ══════════════════
arg = sys.argv[1] if len(sys.argv) > 1 else "12"
true_label = None

if arg.isdigit():                 # 数字：从官方测试集里取第 N 张
    index = int(arg)
    image = read_images("t10k-images-idx3-ubyte")[index]
    true_label = int(read_labels("t10k-labels-idx1-ubyte")[index])
    print(f"来源: 官方测试集第 {index} 张（正确答案 {CLASSES[true_label]}）")
else:                            # 其它：当成图片文件
    image = np.array(Image.open(arg).convert("L").resize((28, 28)))
    if image.mean() > 127:        # 训练数据是"黑底白衣"，白底的照片要先反色
        image = 255 - image
    print(f"来源: 图片文件 {arg}")

# ══════════════════ 3. 转成模型要的张量 ══════════════════
x = torch.tensor(image, dtype=torch.float32) / 255.0   # (28, 28)，0~1
x = (x - MEAN) / STD                                   # 和训练时同样的标准化
x = x.unsqueeze(0).unsqueeze(0)                        # -> (1, 1, 28, 28)

# ══════════════════ 4. 预测 ══════════════════
with torch.no_grad():                                  # 推理不用算梯度
    logits = model(x)                                  # (1, 10) 分数
    probability = torch.softmax(logits, dim=1)[0]      # 变成概率，加起来等于 1

print("-" * 34)
for score, idx in zip(*probability.topk(3)):           # 取概率最大的前 3 名
    print(f"{CLASSES[int(idx)]:<8s} {float(score):>8.2%}")
print("-" * 34)

pred = int(probability.argmax())
print("预测结果:", CLASSES[pred])
if true_label is not None:
    print("是否正确:", "正确" if pred == true_label else f"错误（正确答案 {CLASSES[true_label]})")

