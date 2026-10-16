# 数据和方法来源

- ATFMTraj 作者数据集：https://huggingface.co/datasets/petchthwr/ATFMTraj ，DOI `10.57967/hf/3356`。本次仅使用本地 RKSIa 六个 TSV 与 stats 标签；实际版本以 `reports/data_audit.json` 的文件 SHA256 为准，不能由当前网页版本替代。
- Thaweerath Phisannupawong、Joshua Julian Damanik、Han-Lim Choi. *ATFMTraj: Aircraft Trajectory Classification Data for Air Traffic Management*, 2024。
- 同三位作者，*Aircraft Trajectory Segmentation-based Contrastive Coding: A Framework for Self-supervised Trajectory Representation*. arXiv:2407.20028；正式版 DOI `10.1109/OJITS.2025.3574746`。
- 官方实现：https://github.com/petchthwr/ATSCC ，固定提交 `789ccd72b7496b52e03df7d8599d8c9e6659bdb9`。参考代码的 MIT 许可全文保留在 THIRD_PARTY_LICENSE_ATSCC.md。
- 用户提供 `ITSC_2026_ZhiJunLim.pdf` 第3–4页用于小型骨干、输入/输出归一化和 signed GeM 公式核对；本次未分发该用户提供的论文文件。

作者数据卡说明，RKSI 来自 OpenSky 与 Airportal；坐标已转换为 ENU、按1秒重采样、平滑、除以120km并用NaN补齐。本项目在此基础上恢复米制、每5秒取一点并保留终点。类标签结合 STAR/IAF/跑道；仅用于评估分组，不进入自监督训练。

运行时遥测问题的官方说明：https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_telemetry 。本项目禁用外连的启动器只影响实验进程树，不改变用户全局配置。
