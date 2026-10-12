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
- [已展开的全部源材料](source-materials/README.md)
- [求是平台实机视频（4 分 24.2 秒）](source-materials/00_context/visual_documentation/video_qiushi.mp4)
- [源数据与代码复现指南](reproducibility/SM_Reproducibility_Guide.pdf)
- [数据和复现说明](DATA_AND_REPRODUCIBILITY.md)
- [文件完整性清单](reproducibility/MANIFEST.md)
- [研究谱系](RESEARCH_LINEAGE.md)
- [Zenodo 原始存档](https://doi.org/10.5281/zenodo.19890402)
- [arXiv 论文页面](https://arxiv.org/abs/2604.27092)

Zenodo 中的完整材料现已逐文件展开到 [`source-materials/`](source-materials/)；
读者不必下载或解压总压缩包。全部 2,325 个文件均保留原始五报告结构，可在
GitHub 中分别打开或下载。其中包括源数据、29 个离线 Python 分析脚本、生成图表、
正文数据面板复现、Result 4 原始 CMOS 帧数据、来源映射和逐文件校验值。

其中 2,018 个 NumPy 数组通过 Git LFS 保存，以免普通 Git 历史过度膨胀；在 GitHub
文件页面仍可逐个直接下载。需要完整克隆时，请先安装 Git LFS，或在克隆后执行
`git lfs pull`。

展开后的主要目录为：

| 路径 | 内容 |
| --- | --- |
| `00_context/` | 光学平台与求是系统界面的定性视频记录 |
| `01_result2_paper/` | 初始传输矩阵复现实验 |
| `02_result2_refine_paper/` | 自参考传输矩阵改进实验 |
| `03_result3_paper/` | 主序关系与传输区间实验 |
| `04_result4_paper/` | Complex-B 双线性场、XOR 与语义基准 |
| `05_result4_refine_paper/` | 优势区域和稀疏输出的改进分析 |
| `main_text_data_panels/` | 正文数据面板的审计性重绘 |
| `shared_raw_data/` | Result 4 原始 CMOS 帧级数组 |
| `provenance/` | 文件清单、校验值与图表来源映射 |

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
