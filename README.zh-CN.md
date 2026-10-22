# 真实光学平台上的端到端自主科学发现

[![arXiv](https://img.shields.io/badge/arXiv-2604.27092-b31b1b.svg)](https://arxiv.org/abs/2604.27092)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.19890402.svg)](https://doi.org/10.5281/zenodo.19890402)

[English](README.md) · [运行分析](DATA_AND_REPRODUCIBILITY.md) · [评测方法与结果](EVALUATION.md)

**Qiushi Discovery Engine 将持续自主研究接入真实光学实验。** 系统制定研究方案，
编写并执行实验代码，测量光学响应，再根据结果调整下一步实验。本仓库提供三项
研究的公开测量数据、分析代码和研究记录：任务从复现已有实验，逐步走向将理论
转化为实验，以及从开放主题中形成研究方向。

## 三项研究

| 研究 | 解决的问题 | 实验记录 |
| --- | --- | --- |
| 传输矩阵与波前控制 | 将已有方法迁移到不同的光学平台，并改进聚焦 | 通过 1,025 次标定测量完成 256 × 256 传输矩阵；最佳聚焦增强从 25.59 提升至 46.1 |
| 主序关系与波传输 | 将抽象的排序关系转化为可测量的传输区间 | 从光学测量重建算子，计算并比较区间，进行物理抽查 |
| 光学输入对表示 | 利用相干叠加、散射和强度探测构造相位敏感的交互读出 | 四 token 的 XOR 实验与八 token 的类别标注实验；比较不同特征和读出预算 |

开放式研究持续 206 个研究步骤、1,288.1 分钟，形成 163 份研究记录和 44 个脚本。
三项研究共同展示了一个持续推进的过程：理论、代码与物理测量相互衔接，实验结果
参与决定接下来研究什么、如何研究。

## 光学输入对计算

四个相位的强度测量经过解调和空白扣除，形成依赖两个输入的复数读出；数值计算
保留其完整实部和虚部。由此，线性分类器能够利用相位敏感的输入对信息，并与
单输入强度构成的特征进行比较。

在八 token 的固定输入库实验中，完整复数读出在“所有输入对同时留出一次重复
编号”的测试下，对有序输入对身份、是否同类、有序类别对三项任务均达到 100%
平衡准确率。训练和测试来自同一批采集。[评测说明](EVALUATION.md) 同时列出
未见输入对、类别留出的结果和数字基线，明确每种划分衡量的能力。

## 运行

安装 Python 3.12 与 Git LFS 后，在仓库根目录执行：

```bash
git lfs pull
python -m pip install -r requirements.txt
python -m analysis.result4 --stage main --output outputs/result4
```

当前分析只使用训练数据拟合标准化，保存留出预测，并分别以测量、输入对和类别
作为划分单位。压缩特征需要选择子集时，选择过程放在内层交叉验证中。
[复现指南](DATA_AND_REPRODUCIBILITY.md) 给出稀疏读出、特征预算和四相读出的运行方式。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| [`analysis/`](analysis/) | 当前维护的离线分析与评测代码 |
| [`results/`](results/) | 从公开测量重新计算的精简结果 |
| [`source-materials/`](source-materials/) | 保持原样的 Zenodo v1 存档，包括数据、原始脚本、图表与来源记录 |
| [`paper/`](paper/) | 已公开的论文版本 |
| [`reproducibility/`](reproducibility/) | 原始数据指南与完整性清单 |
| [`tests/`](tests/) | 划分、预处理、特征选择和相位读出的测试 |

原始存档共 2,325 个文件，其中 2,018 个 NumPy 数组使用 Git LFS 保存。
[实机视频](source-materials/00_context/visual_documentation/video_qiushi.mp4) 展示实验环境。
新分析写入 `outputs/`，原始测量和历史结果保留在存档中。光学采集使用独立维护的
仪器控制环境；本仓库提供离线分析。

## 作者与引用

Shuxing Yang, Fujia Chen, Rui Zhao, Junyao Wu, Yize Wang, Haiyao Luo, Ning Han,
Qiaolu Chen, Yuze Hu, Wenhao Li, Mingzhu Li, Hongsheng Chen, and Yihao Yang.

引用格式见 [`CITATION.cff`](CITATION.cff) 和 [`CITATION.bib`](CITATION.bib)。
使用测量数据时，请同时引用论文与数据记录。[研究谱系](RESEARCH_LINEAGE.md) 链接
相关的 Qiushi 研究项目。

[组织仓库](https://github.com/Oxelra-AI/End-to-end-autonomous-scientific-discovery-on-a-real-optical-platform)
· [Shuxing Yang 的个人仓库](https://github.com/Cherish0925/End-to-end-autonomous-scientific-discovery-on-a-real-optical-platform)

## 权利说明

论文、Zenodo 数据与第三方材料沿用已有条款。本仓库不统一声明覆盖全部内容的
许可证；其他使用需求请联系通讯作者。
