# PGSF-Det — 概率几何引导的稀疏融合 RGB-T 小目标无人机检测网络

> 历史 v1 文档归档：以下架构、参数、命令和实验说明保留原文，不代表当前 v2。当前实现与实测结果见项目根目录 README.md。

非配准 RGB-T 无人机检测。数据集 **UAV-CB**，双路独立检测 + 概率几何先验 +
两阶段稀疏跨模态融合。

## 2026-10-10 Bug 修复与验证

修复数据高宽颠倒、原图/网络输入像素空间混用、GT 填充框计入 AP、trainer
验证索引错误、AP 按 IoU 而非分数匹配，以及双模态匹配指标的坐标混用。
AP 现在使用 101 点插值；小目标指标字段改为 `small_recall`，明确其为召回率。

RGB/TIR 均保留完整图像并独立 resize；不再用 RGB 裁剪参数裁剪 TIR。
归一化框不随 resize 缩放，水平翻转使用 `cx'=1-cx`。
翻转矩阵随样本传入模型，几何先验同步更新为 `T A T^-1`。
两方向使用正向/反向 affine；增强残差写回候选所属模态的 P3/P4。
关联监督使用两模态 GT 的同一目标索引，支持乱序候选、多个有效 key 和空匹配。

候选注入排除 padding/空图并保留真实中心；高斯标签中心量化到热力图网格，
保证有正样本。采样和 scatter 使用一致的归一化网格约定。
阶段迁移用 `--ckpt`，同阶段继续训练用 `--resume`；保存/恢复优化器与 AMP 状态。
旧 checkpoint 的 affine buffer 键由加载函数兼容，骨干/检测头缺失仍严格报错。

验证日志在 `outputs/bugfix_validation/`：62 项单元/回归测试通过；
规定输入尺寸的随机前向、反向及 3 步后的有限梯度覆盖通过（907 个参数张量）；
2 张真实训练帧 + 2 张验证帧的两阶段训练、验证、迁移、保存/恢复流程通过。
短流程的未训练验证 AP 为 0，不是性能实验，也不表示小样本过拟合已通过。
历史 `logs/of*.log` 使用旧评测代码，不能作为当前版本的性能依据。

```bash
PY=/home/yangjuanfeng/lab/envs/venv/bin/python
$PY -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=2 $PY scripts/check_forward.py --config configs/pgsf_stage2.yaml \
    --batch 1 --skip-latency --set 'data.rgb_size=[768,1280]'
CUDA_VISIBLE_DEVICES=2 $PY scripts/check_training.py
```

---

## 1. 数据集事实（本地实测，非推测）

`/home/yangjuanfeng/lab/shared/datasets/UAV-CB`，由 `probe_fast.py` 统计：

| 项 | 实测值 |
|---|---|
| 场景数 | 5（building 1191 / cloud 590 / ground 168 / powerline 555 / vegetation 889），共 3393 对 |
| 每图目标数 | **恰好 1 个**，label 恒为 `drone`，两模态都是 |
| RGB 分辨率 | 1920×1080 |
| TIR 分辨率 | 640×512 |
| 框尺寸（RGB 像素） | 中位 68×52，p10 39×26，最小 18×9 |
| 框面积占比 | 中位 1.58e-3（约 3300 px²） |
| **RGB→TIR 归一化中心偏移** | 均值 (0.002, −0.035)，**标准差 (0.042, 0.028)**，p95\|偏移\| (0.080, 0.074) |
| 每场景序列数 | 2（文件名 `<scene>_visible_<seq>_<frame>.jpg` 中的 `1`/`2`） |

**关键结论**：中心偏移存在**系统性垂直位移 + 同量级的逐帧残差**。这正是本设计
不用单一硬仿射配准、而把对应关系建模为二维高斯
`P(p_T|p_R) = N(mu, Sigma)` 的直接依据——硬配准会在相当比例的帧上系统性错位。

### 数据划分

复用仓库内已有的官方口径划分，保证与其他 baseline 可比：

```
/home/yangjuanfeng/lab/projects/SkyGuard-RGBT/data/splits/uavcb/{train,val,test}.txt
```

该划分是论文 6:2:2 分层规则的**本地重建版**（seed 20261009，排除 10 对人工复核
的坏样本，剩余 3383 对：train 2028 / val 678 / test 677）。详见该目录 `README.md`。
RGB 与热红外始终同组。**test 集真值不参与任何先验拟合、模型选择或训练。**

若该目录不存在，`pgsf.data.build_split` 会退回到内部**序列级划分**（每场景序列 2
整体作为 test，序列 1 连续切成 train/val），保证代码在任何环境下都能跑。

---

## 2. 架构

```
RGB B×3×768×1280              TIR B×1×512×640
   │                              │
轻量 CSP 主干                  轻量 CSP 主干            ← 权重独立
   │ P2 64×192×320                 │ P2 64×128×160
   │ P3 128×96×160                 │ P3 128×64×80
   │ P4 256×48×80                  │ P4 256×32×40
   │ P5 384×24×40                  │ P5 384×16×20
   ▼                              ▼
MSCE 小目标显著性增强（P2 上）      同上
   ▼                              ▼
轻量 PAN-FPN（全部统一到 128 通道）
   ▼                              ▼
中心点热力图 → 高召回候选（64/模态）
   │ 48 全局 top-k（局部 NMS） + 16 分区 top-1
   └──────────────┬───────────────┘
                  ▼
      【第一次融合】概率几何引导的中期稀疏跨模态 Transformer
         每候选的稀疏 key 集 = 8 局部可变形采样（带 log P_G）
                           + 16 全局高响应候选（类型偏置）
                           + 1 可学习空匹配候选（类型偏置）
         可靠性门控 γ = σ(MLP([f, z, c_g, u]))，残差 f' = f + γ(z−f)
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
  候选按可微双线性散射写回 P3/P4（权重视图归一化，重叠不叠加）
      ▼                       ▼
  初步检测 → 保留 32 个候选/模态
      └───────────┬───────────┘
                  ▼
      【第二次融合】目标级双向交叉注意力（含空 token + 框先验 log P(b_j|b_i)）
      ▼                       ▼
  DetHead 128→128→5        DetHead 128→128→5     ← 各自独立
      ▼                       ▼
  RGB 检测框                TIR 检测框
```

### 参数量（实测，`check_forward.py` 输出）

| 模块 | 参数 |
|---|---|
| rgb_branch | 11.354 M |
| tir_branch | 11.354 M |
| mid（中期融合） | 0.452 M |
| late（后期融合） | 0.348 M |
| **合计** | **23.51 M** ✅ 预算 50 M 内，低于建议的 25–35 M 区间 |

MACs 174.14 G（FLOPs = 2×MACs = 348.29 G），输入 768×1280 + 512×640。

---

## 3. 核心模块要点

### MSCE（P2 上的残差门控）
四路线索：3/5/7 深度卷积（纹理 / 中尺度 / 上下文）+ 多尺度高斯差分与局部对比度
（`x − blur(x)` 与 `|x − blur(x)|` 并存）。拼接 → 1×1 → Sigmoid 得到显著图 S，
**只作有界残差门**：

```
alpha = alpha_max · tanh(a)      # a 可学习，|alpha| ≤ 0.5
F'    = F ⊙ (1 + alpha · S)
```

绝不用 `S·F` 覆盖原特征——目标很暗时 S 会很小，覆盖等于抹掉有效特征。
单元测试 `test_output_is_not_overwritten_by_saliency` 固化了这一行为。

### 概率几何先验
```
mu    = A [p, 1] + delta(p)      # A 全局仿射（训练集最小二乘拟合），delta 学到的残差
Sigma = L L^T + eps I            # L 下三角，由网络逐点预测，正定
c_g   = sigmoid(H_ref − H(Sigma))      # 几何置信度：Sigma 大 ⇒ 置信低
```

协方差是重点：几何不可信时融合门控会**下调跨模态证据权重**，而不是把错位信息
灌进特征。

注意力中，**几何概率项只作用于局部空间候选**；全局候选与空候选用独立类型偏置。
原因是全局候选的设计目的正是"几何先验失败时的回退路径"，若也用 log P_G 打分
就等于废掉了它。

### 两次融合都零初始化残差
`SparseCrossAttention.w_o` 与 `ff[-1]` 零初始化 → 训练起点上融合是**精确恒等**，
这是 stage-2 从 stage-1 权重安全起步的前提。
副作用是第 0 步这些分支上游拿不到梯度（与 LoRA 的 `lora_B=0` 同机制），
`check_forward.py` 因此做**两段审计**：第 0 步记录预期零梯度项，再跑 3 步优化后
要求**零死参数**。

### 候选生成
- 热力图在 P3 网格（stride 8），CenterNet 高斯目标 + penalty-reduced focal loss。
- 半径由目标**短边**决定：20×20 px 的目标得到紧峰，200×200 的得到宽峰。
- 48 全局（局部 NMS）+ 16 分区（4×4 网格各取 top-1），保证角落目标不被漏。
- 训练时注入 GT 抖动候选，**替换分数最低的预测候选**（候选总数不变）。
- **注入候选在 top-32 选择时无条件保留**——否则训练初期热力图无信息，
  纯按分数排序会把所有正样本丢掉，检测损失永远见不到正样本。

### 检测头输出约定
```
cx = cand_x + tanh(r0)·0.5      # 相对候选点的偏移
cy = cand_y + tanh(r1)·0.5
w  = sigmoid(r2)·max_wh        # 恒正
h  = sigmoid(r3)·max_wh
s  = r4                        # 置信度 logit
```
`decode_pred()` 是**唯一定义该约定的地方**，损失与推理都调它，杜绝训练/推理不一致。

> 这里踩过一个真实的坑：最初 head 输出无约束，导致 `w` 预测为**负数**，CIoU 把
> 面积 clamp 到 eps，梯度消失，框回归静默停摆而 loss 看起来仍然正常（AP 恒为 0）。
> 改成 `sigmoid` 后从构造上消除该失效模式。

### 损失

```
L = L_det^RGB + L_det^TIR + λ_h·L_heat + λ_g·L_geo + λ_a·L_assoc
L_det = focal(cls) + 2.5·CIoU(box) + 0.5·NWD(box)      # A6 可关 NWD
L_heat   = CenterNet penalty-reduced focal，双模态
L_geo    = 高斯 NLL **减去固定参考协方差**（见下），目标按匹配索引 gather
L_assoc  = 同一 GT 的 key 概率质量的负对数；无有效 key 时监督空匹配
```

每个损失都有明确监督源，且至少有一条其他损失到不了的梯度路径。

> `compute_losses` 返回的 `terms` 字典是 **`detach()` 过的记录值**，只用于打印。
> 想单独优化某一项（如只训热力图头做 A/B）必须从 live logits 重新算，
> 否则会拿到 `element 0 of tensors does not require grad`。

> 几何监督的目标必须按"该候选匹配到的目标索引" gather，而不是从 object 0 广播——
> 后者在一帧多于一个目标时就是错的。

### 两个损失权重是实测标定的，不是猜的

**`w_heat: 8.0`** —— 热力图是候选的唯一来源，权重不足会把 AP 直接锁死在 0。
`scripts/heat_ab.py` 在 4 张真实图上跑 400 步的 A/B：

| 变体 | heat@GT | argmax 误差 | 候选命中 | 候选最近距 |
|---|---|---|---|---|
| V1 `w_heat=1` | 0.048 | 52.8 px | 1/2 | 0.098 |
| V2 `w_heat=8` | **0.237** | 47.1 px | **2/2** | **0.0029** |
| V3 只训热力图头 | 0.250 | **15.2 px** | 1/2 | 0.050 |

V3 把 argmax 误差压到 15 px，说明 **backbone/neck 并没有丢掉 tiny drone 的信息**，
从而排除了"特征不足以支撑热力图"这个假设。V1→V2 说明问题就是权重。

**`w_geo: 2.0`** —— NLL 对 `mu` 的梯度约 0.008，CIoU 对框的梯度约 0.060，
**相差约 190 倍**（`scripts/geo_sanity.py`）。若按直觉取 0.1，加权后几何项比框损失
弱两个数量级，等于把几何先验关掉。

### 几何 NLL 的参考基准

二维参考协方差是 `sigma_ref² I`，所以其 logdet 为 `4 log(sigma_ref)`：

```python
logdet_ref = 4 * log(sigma_ref)
return (0.5 * quad + 0.5 * (logdet - logdet_ref)).mean()
```

这是常数平移，只改变记录值，不改变任何梯度。负 NLL 本身不能证明其他损失
被压制；需要检查实际梯度和验证表现。协方差下界用于防止无界收缩。
旧文档中关于常数平移改变各损失优化强度的解释不成立。

---

## 4. 运行

```bash
cd /home/yangjuanfeng/lab/projects/SkyGuard-RGBT/repo/code/PGSF-Det
PY=/home/yangjuanfeng/lab/envs/venv/bin/python

# 形状 / 前向 / 反向 / 参数 / 显存 / 延迟 全量校验
CUDA_VISIBLE_DEVICES=2 $PY scripts/check_forward.py --config configs/pgsf_stage2.yaml --batch 2

# 单元测试（42 个）
$PY tests/test_units.py

# 真实数据小样本过拟合测试（监督链路是否正确的决定性检查）
CUDA_VISIBLE_DEVICES=2 $PY scripts/overfit_test.py \
    --config configs/pgsf_stage2.yaml --n 8 --steps 2000 --batch 4 --lr 3e-4

# 诊断与不变量检查（都可单独运行）
$PY scripts/geo_bound_check.py   # L_geo 是否有下界（抓 sigma 塌缩）
$PY scripts/geo_sanity.py        # NLL 最优点位置与梯度量级
$PY scripts/heat_ab.py           # 热力图瓶颈 A/B（w_heat 标定依据）
$PY scripts/overfit_diag.py 400 5e-4 4   # 训练后逐环节诊断

# 阶段一：单模态检测预训练
CUDA_VISIBLE_DEVICES=2 $PY scripts/train.py --config configs/pgsf_stage1.yaml

# 阶段二：跨模态联合训练（从阶段一权重起步）
CUDA_VISIBLE_DEVICES=2 $PY scripts/train.py --config configs/pgsf_stage2.yaml \
    --ckpt outputs/pgsf_stage1/ckpt/last.pt

# 评测 / 可视化 / 消融
$PY scripts/train.py --config configs/pgsf_stage2.yaml --eval-only \
    --ckpt outputs/pgsf_a6_stage2/ckpt/best.pt --split val
$PY scripts/visualize.py --ckpt outputs/pgsf_a6_stage2/ckpt/best.pt --n 8
$PY scripts/make_ablation.py
```

命令行覆盖：`--set model.neck_out=192 optim.epochs=3 model.use_msce=false`

> 学习率上限：`w_heat=8` 放大梯度后，lr 5e-4 在 AMP 下会出现 loss=NaN。
> **用 3e-4 或更低**。trainer 会跳过非有限损失并累计 `skipped` 计数上报，
> 但这只是兜底，不该依赖它。

---

## 4.1 踩坑：AP 恒为 0 是评测层的假象

这是本项目最隐蔽的一个 bug，值得单独记录。

**症状**：损失正常下降、热力图确实学到了峰（argmax 误差 15 px）、候选能命中，
但 **AP 恒为 0.000**，怎么调都上不去。

**根因**：`inference` 返回的预测框是**各模态原生像素坐标**，而 `evaluate` 把
**归一化**的 `boxes_rgb`（0..1）当 GT 传给了 `eval_detection`。一个 0.5 宽的归一化框
与 640 px 宽的像素框 IoU 恒为 0 —— 指标链路上报的是"完全没检测到"，
但模型其实是对的。

**破案证据**：诊断脚本对同一批预测手算 IoU，top-1 达到 **0.759**（远超 AP50 的 0.5
阈值），而 AP 报 0。既然"预测与 GT 重叠良好"却"AP=0"，问题必然在度量而非模型。

**修法**：
1. GT 改用 `boxes_<modality>_px`（`trainer.evaluate` 与 `overfit_test.evaluate`）
2. `eval_detection` 加坐标空间守卫——像素预测配归一化 GT 直接 `AssertionError`
3. trainer 载入数据后再断言一次 GT 量级

**教训**：**指标卡在某个极值时，先证明评估管线本身是对的**（用手算 IoU 对拍），
再怀疑模型。"AP 恒 0 / loss 恒定 / recall 恒 0"这类症状，几乎总是链路的 bug。

---

## 5. 工程实现要点

- **混合精度**：所有涉及 `linalg.solve` / `solve_triangular` / `logdet` 的地方
  显式升到 float32。AMP 下激活是 fp16 而恒等矩阵是 fp32，`linalg` 拒绝混合 dtype；
  且小协方差在 fp16 下会下溢。
- **归一化坐标**：所有点/框一律 `[0,1]`、顺序 `(x, y)`。`bilinear_grid` 返回
  **归一化**网格坐标——用像素索引会让 `grid_sample` 全部贴边、scatter 越界。
- **散射权重视图归一化**：`scatter_points` 用
  `Σ v_k w_k / (Σ w_k + eps)`，重叠候选得到加权均值而非 3 倍放大。
  注意 `scatter_add_` 索引的是 `self` 的第 1 维，展平轴必须放在第 1 位。
- **多 GPU**：`optim.batch_size` 可调，`train.py` 通过 `--set` 传；判卡后显式绑
  `CUDA_VISIBLE_DEVICES`，长任务用 tmux 保活。
- **参数量分组**：主干 `lr × 0.1`，新初始化的融合模块 `lr × 10`，其余基准 lr。

---

## 6. 消融开关

| 开关 | 字段 |
|---|---|
| A1 MSCE | `model.use_msce` |
| A2 中期 Transformer（无几何先验） | `model.use_mid_fusion` + `use_geo_prior=false` |
| A3 概率几何先验 | `model.use_geo_prior` |
| A4 可靠度门控 + 空匹配 | `model.use_gate` |
| A5 后期目标级 Transformer | `model.use_late_fusion` |
| A6 NWD 辅助定位 | `model.use_nwd` |

`scripts/make_ablation.py` 生成 A0–A6 七个配置，除上述开关外**其余超参完全一致**，
因此 AP 差异可归因到被加入的模块。

---

## 7. 指标

`pgsf/utils/metrics.py`，COCO 口径，双模态分别统计：

- `AP50` / `AP75` / `AP50:95`（101 点插值，IoU 0.5:0.05:0.95）
- `small_recall`：仅对 COCO 小目标（√area < 32 px）的召回
- `P / R / F1`：单阈值协议（UAV-CB 论文风格）
- `match_acc`：两模态 top-1 检测是否命中同一配对 GT；不是 attention 匹配准确率

**SOTA 对比必须使用同一划分与同一协议**（见第 1 节），否则数字没有可比性。
