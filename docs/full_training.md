# 全量训练：GPU0 + GPU7，200 轮预算

训练使用专用环境 `/home/yangjuanfeng/lab/envs/PGSF-Det`。它由 uv 创建，
`include-system-site-packages=false`，Python3.12、Torch2.5.1+cu121、Torchvision0.20.1+cu121。
项目 `pyproject.toml` 和 `uv.lock` 锁定 Linux x86_64 CUDA121 环境；完整安装版本见
`requirements-training.lock.txt`。

```bash
cd /home/yangjuanfeng/lab/projects/SkyGuard-RGBT/repo/code/PGSF-Det
UV_PROJECT_ENVIRONMENT=/home/yangjuanfeng/lab/envs/PGSF-Det uv sync --locked
source /home/yangjuanfeng/lab/envs/PGSF-Det/bin/activate
```

最多 40 轮独立分支预训练 + 160 轮端到端融合训练，合计 200 轮；每阶段验证集
RGB/TIR 平均 AP50:95 连续 30 轮无至少 1e-4 的改善时提前结束。阶段一结束后
阶段二自动加载阶段一验证集最佳 EMA。早停可能使实际轮数少于 200，轮数不会
补回下一阶段。验证结果保存最佳权重；测试集不参与选择。

双卡 DDP 使用每卡独立的训练数据分片、每轮重排、相同学习率调度，梯度同步后更新。
验证集使用不补齐的分片，避免重复计算最后几张图；验证前广播 rank0 EMA 参数和
BN 缓冲，保证两卡评估同一个模型。只有 rank0 写 checkpoint、配置和训练日志。
断点包括模型、EMA、优化器、AMP、epoch 和早停计数，保存时临时文件原子替换。

数据包含全部 2028 对训练帧、678 对验证帧；固定划分是仓库的本地重建协议，不
宣称官方 SOTA 协议已经核验。677 对测试帧保留，在本次训练流程中不做评估。

## 增强策略

该数据集单目标、目标较小、两模态没有精确配准。使用有界的空间和外观扰动：

| 增强 | 阶段一 | 阶段二 |
|---|---|---|
| 同步水平翻转 | 0.5 | 0.5 |
| 同步真实画布缩放 | 概率 0.5，倍率 0.90–1.15 | 相同 |
| RGB 颜色增益扰动 | ±15% | ±15% |
| TIR 增益扰动 | ±10% | ±10% |
| 各模态独立平移 | 概率 0.15，范围 ±1.5% | 概率 0.25，范围 ±2.5% |
| 单侧退化 | 概率 0.05 | 概率 0.10 |
| 单侧模态丢弃 | 关闭 | 概率 0.03 |
| 局部几何先验扰动 | 关闭 | 概率 0.10，std=0.015 |
| 后期减弱增强 | 最后 10 轮 | 最后 20 轮 |

画布缩放改变目标尺寸，中心缩放后裁剪或按图像均值补边；此前重复 resize 的伪缩放
在本次配置中替换。各模态实际像素变换同步更新归一化框和仿射关系。被裁剪目标
保留至少 50% 可见面积才能继续检测监督；不完整的框不参与几何/尺度 NLL，防止
裁剪中心偏移造成错误几何标签。验证时所有随机增强关闭。

退化随机选择 RGB 模糊（sigma0.2–0.8）或 TIR 轻噪声（std6/255），每次保留另一个
模态；避免同时破坏两侧。丢弃模态不接受伪背景检测监督，而接受空匹配监督。
后期关闭独立平移、退化和丢弃，缩放收窄到 0.96–1.04，RGB 颜色扰动减半。
几何和标注正确同步，不做 GT 依赖的裁剪。

## Batch 与日志

正式 batch 在 `scripts/benchmark_ddp_batch.py` 中用 GPU0/GPU7 依次实测选取。
记录在 `outputs/fulltrain_preparation/batch_*.json`，速度统计包括 DataLoader 等待、
增强、H2D、前后向、DDP、梯度裁剪、AdamW 与 EMA；warmup 不计入。以双卡的样本/秒
为主，显存为次；每个 batch 在独立进程测量，避免同时占卡干扰。

| 每卡 batch | 双卡总 batch | 实测对图像/秒 | 最大峰值显存 GiB |
|---|---|---:|---:|
| 4 | 8 | 22.06 | 3.97 |
| 6 | 12 | 31.61 | 5.82 |
| 8 | 16 | 41.11 | 7.68 |
| 10 | 20 | OOM | 进程使用约 9.52 |

因此正式使用每卡 **8**、总 **16**，保留约 2 GiB 显存余量；继续以原始基准 lr2e-4
训练，不因 batch 增加盲目线性放大学习率。短吞吐测量取预热后的 12 步，batch8
另做 30 步较长复核，见 `batch_8_long.log`。

日志为训练表格进度条，列为 Epoch/GPU_mem/box_loss/cls_loss/nwd_loss/heat_loss/
fusion_loss/Instances/Size(R/T)。box、cls、NWD、heat 是 rank0 本轮两模态平均原始
分项，用于可读性；fusion_loss 是加权的几何/关联/尺度损失和（合法 NLL 可为负）。
Instances 是 rank0 当前批次两侧有效目标数；GPU_mem 为 rank0 峰值 allocated GiB。
验证表格分别打印 RGB、TIR 和 mean 的 Images/Instances/P/R/mAP50/mAP50-95。
详细全部损失与双卡聚合结果在各阶段 `train.log` 的 JSONL 中。

## 启动与恢复

```bash
/home/yangjuanfeng/lab/envs/PGSF-Det/bin/python -u scripts/run_full_training.py
```

编排器固定 `CUDA_VISIBLE_DEVICES=0,7`、使用两个 DDP 进程，生成
`outputs/fulltrain_20261010_200e/status.json` 和 `stage1_console.log` /
`stage2_console.log`。失败后显式再次执行同一命令会从现有 last.pt 恢复；已完成阶段
有 `completed.json` 标记会跳过。不要在已有进程运行时重复启动。

```bash
# 实时查看训练
 /home/yangjuanfeng/lab/envs/PGSF-Det/bin/python scripts/watch_training.py
# 单次查看最近 5 轮
 /home/yangjuanfeng/lab/envs/PGSF-Det/bin/python scripts/watch_training.py --once --last 5
# 原始日志
 tail -f outputs/fulltrain_20261010_200e/stage1_console.log
# 进入保活会话
 tmux attach -t pgsf-full200
```

每阶段最佳 checkpoint 为 `stageN/ckpt/best.pt`，最新可恢复 checkpoint 为
`stageN/ckpt/last.pt`。正式指标随 epoch 写入，不使用小样本调试权重初始化。

查看器只读日志，不导入 torch、不占 GPU。终端中默认每 0.2 秒检查并刷新
刷新最近 3 轮；切换阶段时自动读取两阶段日志，Ctrl+C 只停止查看器。实际训练
日志以 0.2 秒为最小间隔逐批更新，查看器也每 0.2 秒刷新。`--interval` 可调整检查间隔；
非交互输出会追加变化后的快照，`--once` 适合脚本调用或保存文本。`mean` 是
RGB/TIR 指标的算术平均，不是把两侧预测合并后的 pooled AP。

表格按终端宽度自动计算列宽，标题与数值右对齐；窄窗口按列分组，进度条单独
显示。窗口尺寸变化会实时重排，高度不足时优先显示当前轮次。`--width 80` 可
强制宽度；没有底部 Live 状态行。逐行清空重绘避免旧文字残留，交互模式使用终端
备用屏幕，退出时恢复原屏幕。升级查看器后需 Ctrl+C 并重新运行，
已启动的 Python 查看器不会自动加载新代码。
