# 真实光学平台上的端到端自主科学发现

[![arXiv](https://img.shields.io/badge/arXiv-2604.27092-b31b1b.svg)](https://arxiv.org/abs/2604.27092)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.19890402.svg)](https://doi.org/10.5281/zenodo.19890402)

本仓库完整汇集论文 **《End-to-end autonomous scientific discovery on a real
optical platform》** 的论文、源数据、离线分析代码、图表、来源记录与复现材料。

这项工作提出并验证了 **Qiushi Discovery Engine**：一个能够在真实自由空间
光学平台上开展长程研究的自主科研系统。三个递进研究分别检验了系统复现已发表
实验、把抽象理论转化为物理实验，以及从开放主题出发提出并验证新物理机制的能力。

## 三项核心研究

1. **传输矩阵实验复现。** 系统把已发表方案迁移到不同的光学平台，完成
   256 × 256 传输矩阵与 1,025 次标定测量，并将最佳聚焦增强从 25.59 提升到 46.1。
2. **从理论到实验。** 系统把波相干性的主序关系转化为可测的传输响应区间，
   在真实平台上验证了可比谱之间的区间嵌套关系。
3. **开放式科学发现。** 系统从“面向人工智能的光计算”这一宽泛主题出发，提出
   光学双线性交互：相干叠加、散射与平方律探测共同产生依赖于两个输入的复数场，
   其结构对应 Transformer 注意力中的双线性相容性计算。

开放式研究共经历 206 个 Agent Step、1,288.1 分钟、1.459 亿 token、3,242 次
模型调用和 1,242 次工具调用，形成 163 份研究记录和 44 个脚本。结论来自真实
物理测量，而不是纯模拟流程。

## 完整材料

- [论文 PDF](paper/End-to-end_autonomous_scientific_discovery.pdf)
- [源数据与代码复现指南](reproducibility/SM_Reproducibility_Guide.pdf)
- [数据和复现说明](DATA_AND_REPRODUCIBILITY.md)
- [文件完整性清单](reproducibility/MANIFEST.md)
- [研究谱系](RESEARCH_LINEAGE.md)
- [Zenodo 原始存档](https://doi.org/10.5281/zenodo.19890402)
- [arXiv 论文页面](https://arxiv.org/abs/2604.27092)

完整的 1.4 GB 源材料包以 `SM_Source_Materials_CORE_5REPORTS.tar.gz` 的原始形式
保存于 GitHub Release `v1.0-source-materials`。其中包括五组报告、源数据、29 个
离线 Python 分析脚本、生成图表、正文数据面板复现、Result 4 原始 CMOS 帧数据、
来源映射和校验值。

## 可复现范围

公开材料支持检查数据、代码和离线重跑分析。重新采集光学测量仍需要原始实验平台
及其标定后的仪器控制环境；Qiushi Engine 核心系统与硬件控制软件不在本次公开范围
内。论文与正式补充材料是科学论述的权威版本，源材料包负责数据、代码与来源追溯。

## 引用

引用信息见 [`CITATION.cff`](CITATION.cff) 和 [`CITATION.bib`](CITATION.bib)。
使用存档数据或分析代码时，应同时引用 arXiv 论文和 Zenodo 数据记录。

## 权利说明

本仓库不统一声明一项覆盖全部内容的许可证。arXiv 论文与 Zenodo 记录分别遵循其
已有条款，第三方材料继续遵循各自权利约定；其他使用需求请联系通讯作者。
