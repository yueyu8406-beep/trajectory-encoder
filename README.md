# 轨迹 Encoder

按 [PLAN.md](PLAN.md) 训练 ATSCC/ITSC-inspired Encoder。当前状态：**完整对照训练中**。四组超参数筛选已完成，各30轮；验证集选择 ε=120m、τ=.01。8维三个seed已完成，4维/16维对照正在继续，最终TEST尚未评分。

已经完成的8维 seed42 候选权重将经校验发布为 [`checkpoints/d8_e120_t0p01_s42.pt`](checkpoints/d8_e120_t0p01_s42.pt)。这是候选版本，最终维数和代表seed仍须等待完整验证集比较。当前候选几何探针分数低于随机初始化基线；详细结果如实保留，复杂度用途仍待验证。

数据：ATFMTraj RKSIa，训练/验证/测试 6877/1214/8092 条。原始 TSV 不修改、不上传。新项目不复用上一轮权重或缓存。独立虚拟环境使用已装 CUDA PyTorch，不改系统环境。

模型：9维输入、4层 causal Transformer、宽256、FFN512、4头、dropout .35、20%输入掩码、RDP flat-SNN。比较温度 .01/.05、RDP ε=120/1200m；再比较 4/8/16 维各三随机种子。sum 为固定主 pooling，mean / signed GeM-p2 为敏感性分析。

详细选择与评估规则见 [PROTOCOL.md](PROTOCOL.md)。最终 TEST 不参与参数选择。当前 TSV 缺绝对时间与航班 ID；交付范围为 Encoder，小时场景验证不冒充完成。

## 复跑

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m trajectory_encoder.data --source 'E:\樟宜机场项目\hunging face数据' --out prepared
.\.venv\Scripts\python.exe -m trajectory_encoder.checks
.\.venv\Scripts\python.exe -m trajectory_encoder.benchmark
.\.venv\Scripts\python.exe -m trajectory_encoder.pipeline
.\.venv\Scripts\python.exe -m trajectory_encoder.finalize
```

依赖 Python3.10、PyTorch2.7.1+cu128、numpy2.2.6、pandas2.3.3；本机 RTX5060 Laptop 8GB。B16/bf16 经真实训练步与最长批次测试，未用梯度累积替代大 batch。

## 加载接口（权重完成后可用）

```python
from trajectory_encoder.inference import FrozenEncoder
encoder = FrozenEncoder('encoder/encoder.pt', pooling='sum', device='cpu')
out = encoder.encode(enu_m, resample_from_1hz=True)
# enu_m: 1Hz RKSI本地ENU米制坐标，N×3
print(out['tokens'].shape, out['embedding'].shape)
```

采用 [ATSCC](https://github.com/petchthwr/ATSCC/tree/789ccd72b7496b52e03df7d8599d8c9e6659bdb9) 的指定提交为实现参考；许可见 [THIRD_PARTY_LICENSE_ATSCC.md](THIRD_PARTY_LICENSE_ATSCC.md)。几何指标是表示探针，不是人工复杂度真值。
