# Encoder 实验协议（训练前固定）

目标是交付 PLAN.md 的自监督轨迹 Encoder。此目录完全新建；不读旧脚本、缓存、日志或权重。原始 TSV 只读。独立 `.venv` 通过 system-site-packages 使用已有 PyTorch/CUDA，不修改原环境。

## 数据

使用本地 Hugging Face ATFMTraj RKSIa 数据：官方 TRAIN 8091 条，TEST 8092 条。TRAIN 中按几何签名分组划出 15% 验证集，最终 train/val/test = 6877/1214/8092。先做全坐标精确哈希和 32 点、50 米网格形状签名的跨集检查；签名只是部分重复筛查，不能证明不存在其他近重复。没有日期、航班 ID，因此不能验证真实航班/日期级隔离；不伪造这些字段。

每条轨迹是 1 秒采样的相对坐标，恢复 `r_max=120000m` 后每 5 秒取一点并保留终点；此为减少冗余、适配 8GB GPU 的预处理选择，源轨迹未裁剪。数据文件、样本索引、预处理参数均保存指纹。NaN 只作尾部 padding；9 维特征的均值/标准差只在训练 token 上拟合。方向使用前向差分，最后一项重复；因此特征本身有下一采样点的信息，Transformer 在给定特征上是因果的。

## 模型与损失

9→256 输入投影、L2、4 层 causal Transformer、256→D 输出投影和逐 token L2。4 heads，FFN 512，dropout=.35，无显式位置编码，不增加最终 LayerNorm。D=4/8/16。保留 ATSCC 指定提交的无 bias 注意力投影、sqrt(model_dim) 缩放和 norm/residual 顺序。

真实二项输入掩码概率为 .2；输入被遮蔽的非 padding token 仍参与监督，与 ATSCC 的监督索引一致。全遮蔽前缀的注意力返回零，不允许其读取未来。损失的正例是同一轨迹中同一 RDP 段的其他 token，负例是其他段/其他轨迹的 token；排除自身与 padding。单点段不能作锚点，但可作负例。不同轨迹的局部段编号绝不合并。采用 `logsumexp(negative/τ)-logsumexp(positive/τ)`，flat loss 无层次项/复杂度标签。数学排除不使用乘零后仍进入 exp 的方式。

RDP 在米制三维坐标计算。epsilon=120/1200m（原归一化尺度 .001/.01），温度 .01/.05：先完成 D=8、seed42 的 2×2 筛选，各最多 30 epoch；再对选出的超参完成 D=4/8/16、seed42/43/44，已完成的 D8 seed42 直接计入。总计 12 个独立完整训练。该有限网格是预算内的计划实现，未宣称穷尽论文所有 epsilon/temperature。

AdamW lr=1e-5, weight_decay=1e-5；梯度裁剪1；bf16 混合精度。实测 B8/B16 的吞吐和最长轨迹显存后选择真实 batch，不通过梯度累积冒充 batch64。若容量允许，可在同一 GPU 并行两个独立实验；各自独立模型和优化器。

运行说明：原 PowerShell 进程在第 1 轮后因 Visual Studio/Application Insights 遥测连接被自动审批终止。已检查训练代码无网络调用，并增加 `run_offline.py` 与 `offline_guard/sitecustomize.py`，仅在实验子进程中禁止 Python DNS/外连、设置遥测 opt-out，不修改全局环境。验证阻止 DNS/连接、本地 CUDA 及 70 秒进程存活后，从保存的模型/优化器/随机状态续跑。实际运行通过不包含 PowerShell 的进程树启动；这次中断不改变实验参数和数据。后续可用 `python run_offline.py pipeline` / `python run_offline.py finalize` 在同样限制下复跑。

并行资源回退：第一次两 worker 运行期间，一个 worker 在 epoch20 后超过5分钟不推进，另一个正常完成30轮；为保证可完成性，仅停止本实验进程树，保留 checkpoint，随后将实际并发数降为1并续训。模型、种子、batch、数据、损失和选择规则均不变。`benchmark.json` 保留最初实测与实际回退，`runtime_events.json` 记录运行事件。

## 选择与评估

单个 run 以固定验证 batch 的最低验证 flat-SNN loss 选 checkpoint；不直接跨温度比较 loss 数值。筛选 epsilon/temperature 用验证集 ridge probe 的四个目标 R²（总路径长度、累计转角、曲折度、时长），分别计算全局和 STAR 内宏平均，8 项等权平均。每个目标的 ridge alpha 只用验证集选择，特征和目标缩放只拟合训练集。

最终维数以三 seed 的上述验证分数均值选择；精确同分时选较低维数。主 pooling 预先固定为 sum，mean 和论文 signed GeM-p2 为敏感性分析，pooling 后不另做 L2。代表权重选择所选维数中验证分数居中的 seed，不挑测试最佳 seed。所有选择写入 selection.json 后才进行最终 TEST 评估。

同时保留随机初始化 Encoder 基线；报告逐维方差、有效秩、成对距离、范数与长度相关、长度分层和 STAR 内 probe。所有几何目标只是可计算的表示探针，不能称为人工复杂度真值、holding 标注或真实小时场景验证。因当前数据缺绝对时间，相关维数/小时点云验证未执行，交付状态仅限 Encoder。

这是针对当前数据和计划改造的 ATSCC/ITSC-inspired Encoder，不是原论文数据集上的严格复现。具体代码差异、失败情况和实测结果随交付保存。
