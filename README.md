# DRAG 与 BR-DRAG 论文复现

依据 Xiao 等人在 IEEE TIFS 发表的论文第 III、IV、VI 节实现。训练采用单进程 PyTorch 仿真。每轮从 40 个客户端中均匀无放回抽取 10 个，客户端运行 5 步 SGD，批大小 10，学习率 0.01。实验按类别做 Dirichlet 划分，默认比较 β=0.1 和 β=0.5。EMNIST Balanced 运行 3000 轮，CIFAR-10/100 运行 4000 轮。 [论文 DOI](https://doi.org/10.1109/TIFS.2026.3707437) · [作者预印本](https://arxiv.org/abs/2601.06903)

## 安装和验证

使用 Python 3.9–3.12。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe reproduce.py --figure smoke --threads 2
```

下文以已激活环境中的 `python` 为例。已有可用的 PyTorch 和 torchvision 环境时，可以直接运行。论文使用 PyTorch 2.5.1 和 RTX 4090。CUDA wheel 的安装命令见 [PyTorch 2.5.1](https://pytorch.org/get-started/previous-versions/#v251)。

## 运行论文实验

```powershell
# 列出每幅图的运行次数和通信轮数
python reproduce.py --figure all --plan

# 图 4 和图 12。默认自动检测 CUDA
python reproduce.py --figure 4 12 --download --output results/paper

# 一个 BR-DRAG 设置。图 12(b) 中对应的条件
python train.py --dataset cifar10 --method br-drag --attack sign-flip --attacker-ratio 0.3 --beta 0.5 --download --output results/br-drag-cifar10

# 多种子。图中曲线为均值，阴影为一个标准差
python reproduce.py --figure 12 --seeds 0 1 2 --download --output results/multiseed

# 短跑，检查真实数据入口和绘图流程
python reproduce.py --figure 4 --rounds 20 --eval-every 5 --download --output results/shortrun

# 从相同输出目录和种子重画已跑完的图
python reproduce.py --figure 4 --plot-only --output results/paper
```

`--figure all` 按图执行图 3–22。单个种子共 206 次运行、812,000 个通信轮次。`--download` 交给 torchvision 下载并校验数据。也可使用 `--data-dir` 指向已有数据，CIFAR-10 目录应包含 `cifar-10-batches-py/`，CIFAR-100 应包含 `cifar-100-python/`，EMNIST 应包含 `EMNIST/raw/`。若官方 CIFAR 下载较慢，可以将校验过的原始压缩包放入数据目录，再带 `--download` 运行以解包。 [CIFAR-10/100 数据源与校验值](https://www.cs.toronto.edu/~kriz/cifar.html) · [EMNIST 数据源](https://www.nist.gov/itl/products-and-services/emnist-dataset)

本目录的 `data/` 已准备好三个数据集。CIFAR 压缩包使用 PaddlePaddle 镜像，校验官方 MD5。EMNIST Balanced 的四个 IDX 压缩文件使用固定版本镜像并校验 SHA-256。 [CIFAR 镜像来源](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/dataset/cifar.py) · [EMNIST 镜像版本与校验值](https://huggingface.co/datasets/Heliosoph/EMNIST/commit/e1ce0ec731a6482d8a708c641c2030d6a6cc1f31)

| 图 | 实验设置 |
| --- | --- |
| 3–6 | EMNIST、CIFAR-10、CIFAR-100 CNN、CIFAR-100 WRN-28-4 上的 DRAG 与 5 个基线 |
| 7–9 | CIFAR-10 上的参与客户端数、α、c 敏感性 |
| 10 | CIFAR-100 上的良性方向范数比和余弦诊断 |
| 11–16 | 30% 拜占庭客户端下的 BR-DRAG 与 4 个基线 |
| 17–19 | 60% 拜占庭客户端下的噪声注入、符号翻转与标签翻转 |
| 20 | 服务器用时与平均客户端用时之比 |
| 21 | 根数据集样本数、标签污染比例、类别缺失消融 |
| 22 | BR-DRAG 的 DoD 修正与 clipped-cosine 替换消融 |

每次运行写入 `config.json`、`split.json`、`metrics.csv` 和 `complete.json`。准确率以百分比记录，默认每 10 轮评估一次，并在第 0 轮和最后一轮评估。可用 `--eval-every 1` 逐轮评估。数值失稳会记录发生轮次，曲线截在最后一个有效测试点。批量运行会跳过已完成且配置相同的任务。中断后，可移动未完成的单次运行目录，再重跑同一命令。

## BR-DRAG 与仅服务器根数据训练的对照

两组每轮都从各自当前的全局参数 θₜ 出发，在服务器持有的 3000 条干净根数据样本上运行同样的 5 步 SGD，得到参考更新 rₜ。原版 BR-DRAG 用 rₜ 修正并聚合客户端上传的更新。对照组 `root-only` 直接采用 θₜ₊₁ = θₜ + rₜ，只执行服务器 SGD。带 BatchNorm 的模型使用服务器 SGD 后的缓冲区。

预设沿用图 12 的 CIFAR-10 设置，BR-DRAG 面对 30% 客户端符号翻转，并比较 β=0.1 和 β=0.5。每对实验共用种子、初始模型、根数据、根数据的批次采样顺序和测试集。`root-only` 的模型更新只依赖服务器，因此同一种子的曲线在两个 β 下相同。

```powershell
# 查看计划，跳过训练
python reproduce.py --figure br-drag-root-only --seeds 0 1 2 --plan

# 运行对照并绘图
python reproduce.py --figure br-drag-root-only --seeds 0 1 2 --output results/root-only
```

默认每次运行 4000 轮。准确率对照图位于 `results/root-only/br-drag-root-only/br-drag-root-only.png`，各组的 `metrics.csv` 记录测试损失和参考更新范数。轮数对齐了服务器 SGD 的步数。对比收敛速度和最终准确率，可以衡量在这组攻击条件下，纳入客户端更新对 BR-DRAG 的净影响。需要单独运行服务器对照组时，使用 `train.py --method root-only`。

## 与论文设置的对应关系

- **DRAG** 遵循 Algorithm 1。第 0 轮先做一遍本地 SGD 来估计参考方向，再从同一全局模型出发独立采样进行正式更新。后续参考方向按式 (5b) 更新，α=0.25。β=0.1 时 c=0.25，β=0.5 时 c=0.1。
- **BR-DRAG** 遵循 Algorithm 2 和式 (13)–(16)。服务器每轮在根数据上运行同样的本地 SGD，按参考方向的范数缩放上传更新并做 DoD 修正，默认 c=0.5。公式中的零范数采用显式边界约定。DRAG 在零参考方向时保留原更新，BR-DRAG 将零参考或零上传映射为零。
- **数据与网络**。论文给出了卷积层数及核尺寸，CNN 的具体通道数、隐藏层宽度、预处理和随机种子采用了本实现的设置。CNN 通道依次为 32、64，CIFAR-100 增加 128 通道层。全连接隐藏层为 128。输入按数据集标准化，使用原始训练与测试划分。WRN 对各客户端浮点 BatchNorm 缓冲区取均值，计数取最大值，拜占庭扰动作用于模型参数增量。默认种子为 0。
- **根数据**。默认从基准训练集分层抽取 3000 个样本，并从客户端池中移除。论文的 confident learning 与异常筛查参数和留样索引未公布。若有审查后的索引，可通过 `--vetted-indices` 传入每行一个、从 0 开始的原始训练索引。图 21 固定 6000 个后备根样本，供不同根数据设置取子集，使客户端划分一致。图 21(a) 的图注写 CIFAR-10，正文写 CIFAR-100，预设采用图注的 CIFAR-10。图 21(b) 用随机改错标签实现根数据污染。
- **扰动与诊断**。噪声注入按第 VI 节解释为 Gaussian 标量乘单个客户端增量，N(0,3) 中 3 按方差处理。标签翻转在被选中的恶意客户端训练分区中预先固定一半标签，映射为 L−1−label。Min-Max 和 Min-Sum 使用原文距离约束，论文未给出的扰动方向采用良性均值的负符号方向。图 10 采用无攻击条件，因为其图注没有标明攻击类型。
- **基线**。FedAvg 按参与客户端等权聚合。SCAFFOLD 采用论文指定的首批梯度控制变量。FedACG 采用原算法的前瞻初始化和服务器动量。FLTrust 使用归一化信任权重，图 22 的对照沿用 BR-DRAG 的参与客户端数分母。原文中的 RFA 和 RAGA 分别对模型和增量求几何中位数，两者目标平移等价。本实现分别采用有限步平滑 Weiszfeld 和带收敛容差的近似求解。 [FedACG 原文](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Communication-Efficient_Federated_Learning_with_Accelerated_Client_Gradient_CVPR_2024_paper.html) · [RFA 原文](https://arxiv.org/abs/1912.13445) · [RAGA 原文](https://arxiv.org/abs/2403.13374)

图中的硬件计时依赖当前设备及其负载。实现选择、随机划分及运行环境会影响准确率和收敛曲线。合成分类任务仅用于验证端到端训练、数值失稳处理和绘图。
