#极简版

## 怎么跑

```powershell
cd D:\pytorch_project\fashion_mnist_project\simple_version

python train.py        # 训练，约 9 分钟，权重存到 outputs\simple_cnn.pth
python predict.py      # 预测官方测试集第 12 张
python predict.py 3    # 预测官方测试集第 3 张
python predict.py D:\照片\鞋子.jpg
```

## 两个文件分别在干什么

| 文件 | 内容 |
|---|---|
| `train.py` | 读 IDX 数据 → 定义 4 层 CNN → 训练 → 保存权重，从上往下一条线 |
| `predict.py` | 加载权重 → 取一张图 → 前向 + softmax → 打印前 3 个类别 |

## 和完整版的区别

| | 极简版 | 完整版（项目根目录） |
|---|---|---|
| 文件数 | 2 个 | `src/` 里 6 个模块 + 3 个入口脚本 |
| 参数设置 | 文件开头几个常量 | 命令行参数 + 配置文件 |
| 数据 | numpy 直接转张量 | 自定义 `Dataset` + `transforms` |
| 模型 | Conv + ReLU + Pool + 全连接 | 额外有 BatchNorm、Dropout、Kaiming 初始化 |
| 训练技巧 | 只用 Adam | 余弦退火、早停、随机种子、训练曲线 |
| 输出 | 终端打印 | 权重、日志、曲线图、混淆矩阵、预测结果图 |
| 测试准确率 | 见下 | 93.03% |

## 极简版的结果

训练 10 轮（CPU 约 9 分钟），测试集准确率 **91.94%**（完整版 93.03%）。

每轮的记录：

```
第  1 轮  训练 loss 0.4848  训练准确率 0.8234  验证准确率 0.8757
第  5 轮  训练 loss 0.1745  训练准确率 0.9371  验证准确率 0.9227
第 10 轮  训练 loss 0.0772  训练准确率 0.9711  验证准确率 0.9228
测试集准确率: 0.9194
```

少的那 1.1 个百分点，就是 BatchNorm、Dropout 和更多训练技巧带来的。
想补回来很简单，在 `CNN.__init__` 里加两行：

```python
self.bn = nn.BatchNorm2d(32)      # 每层卷积后面加
self.dropout = nn.Dropout(0.3)    # 全连接前面加
```

