# PGSF-Det 当前架构的文生图提示词

依据 2026-10-10 修复后的 v2 代码。图示为完整阶段二网络；阶段一训练时关闭两次
融合和条件几何上下文，仍保留独立候选、区域采样和检测头。仅复用预训练主干结构，
双路权重独立；图中尺寸均为 H×W。

## 可直接复制的提示词

绘制一张计算机视觉论文风格的神经网络架构图，标题为
“PGSF-Det: Probabilistic Geometry-guided Sparse Fusion for Unregistered RGB-T Detection”。
白色背景，清晰的二维矢量图，横向宽幅画布，整体从左到右。RGB 分支在上，TIR
分支在下，两个分支对齐。RGB 使用低饱和蓝色，TIR 使用低饱和橙色，几何模块紫色，
融合模块浅绿色。细线、箭头、统一字体，模块内只放短英文标签，文字清晰。

左侧输入是同一无人机的 RGB 和灰度热红外图像，两张图中的目标位置有明显偏移。
标签为 “RGB: 3 × 800 × 1280” 和 “TIR: 1 × 512 × 640”。两路独立的
“YOLO11s Backbone”，明确写 “Independent weights”。每路输出四个特征层
P2/P3/P4/P5，主干通道数分别是 128/256/256/512。P2 上使用 “LiteMSCE”，
旁边小图展示 3×3、5×5、7×7 深度卷积和 Gaussian Difference，并用残差门控
“F′ = F ⊙ (1 + αS)” 增强；P3/P4/P5 直接进入下一模块。

随后每路使用 “Lightweight Multiscale Neck”，输出 P2/P3/P4/P5，各128通道。
每路 P2/P3 接 “Dense Proposal Head”，分三个小输出：Center Heatmap、Coarse Box、
Box Quality。候选模块标 “64 candidates: 32 P2 + 16 P3 + 16 spatial supplements”。
附短标签 “Center-objectness ranking”，不得把框质量画成热力图候选筛选的乘法门。

在两路中间上方放置 “Both-image Context”：两路 Neck 的 P5 全局平均池化、拼接、
MLP，得到128维上下文。它与源候选特征、中心和粗框尺寸一起进入
“Directional Conditional Geometry”，绘制分开的 R→T 和 T→R 小预测器。模块输出
“μ = A p + b + Δ” 和 “Σ ≻ 0”。全局 affine 旁放小虚线标签
“Fitted on training pairs only”。用概率椭圆表达对应位置不确定性，不画图像对齐或扭曲。

中央第一个大模块为 “Mid Sparse Cross-modal Transformer”。显示两方向的交叉箭头。
每个源候选访问对侧模态：8个协方差引导的局部可学习采样点，其中4个在P2、4个在P3；
16个全局高响应候选；1个可学习空匹配 token，合计25个 key。小插图用一个椭圆、
8个采样点、16个远处候选及一个空心 null 标记说明局部搜索、全局回退和空匹配。
标 “128-D, 4 heads” 与 “Reliability-gated residual”。几何偏置只引导局部采样，
全局候选与空匹配作为回退。可放简短公式 “f′ = f + γΔf”。

第一次融合后重新分成 RGB/TIR 两路，分别标
“Normalized Residual Scatter to Own P3/P4”。增强后的候选仍在本模态坐标系；
各自P2/P5保留。不要把RGB残差散射到TIR特征图。

每路随后画 “Top-32 + Multiscale Region Sampling”。在粗框区域内每尺度取9个点，
从本模态 P2/P3/P4/P5 提取特征，max/mean pool 后投影。显示
“Enhanced candidate + ROI feature + Box encoding”，形成 “32 × 128 region tokens”。
框用于区域采样，不画成固定大小窗口，不画成正式RoIAlign。

第二个中央大模块为 “Late Region-level Cross-modal Transformer”，两路32个区域token
双向交互，内部短标签 “Semantic matching”“Center prior”“Learned size relation”
“Null match”“Reliability gate”。用虚线从第一个几何模块连接到此模块，表示复用中心
分布μ/Σ；从Both-image Context连到尺度关系预测器。不画强制一一匹配。

最后重新分成两个 “Independent Refinement Head”，各标 “128 → 128 → 5”。
每路输出相对粗框的中心/宽高残差和置信度，经过 “Decode + Score fusion + NMS”，
得到 “RGB Boxes” 和 “TIR Boxes”，框画在各自原图上，位置和尺寸可以不同。
角落添加小图例 “Solid: feature flow; Dashed: geometry/context”。

整张图强调三点：未配准双模态输入、概率几何引导的稀疏中期融合、区域级后期融合。
不要加入性能数字、SOTA 标记、图像级硬配准、共享主干、DETR decoder、第三次融合、
Mosaic、DFL、光流、跟踪模块或推理时的GT框。两阶段训练只用底部小字
“Stage 1: independent detection; Stage 2: joint fusion”，不要将训练阶段画成两次推理。

## 图形校核要点

- Neck 在第一次跨模态融合之前执行，P3/P4 残差写回后通过区域采样进入最终检测，
  不额外画第二套 Neck。
- 主干 P2/P3/P4/P5 通道：128/256/256/512；Neck 全部128。
- 默认各64候选，中期每query的key预算8+16+1，后期各32区域token。
- quality 保留作监督和融合证据，不控制候选召回；最终置信度是精修质量 sigmoid
  乘中心 objectness 的平方根。
- 默认最终候选只有32个，max_det=50只是上限；要实际输出50个候选需改后期token预算。
- 生成图中文字和箭头须人工核对；严格数据流见 architecture_current.mmd。
