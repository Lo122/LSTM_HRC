# Step 监督方式 A/B/C 消融：实施前检查（历史记录）

下文保留实施前发现与当时的待定项。用户已确认的最终协议、实现和运行方式以本目录 README.md 为准。

日期：2026-09-29。当前仅建立实验目录并记录检查结论；未新增训练代码、未修改现有训练流程、未启动训练。

## 1. 已确定的实验范围

- 主干保持原方案的 LSTM；本轮先比较 step 监督方式，暂不做特征选择或 optimizer/lr 搜索。
- 约 4 秒历史：原始 30 fps，每隔 3 帧采样一次，共 40 点。首末采样点相隔 117 帧，即 3.9 秒，覆盖 118 个原始帧位置。
- A/B/C 使用相同的 15 折 LOSO、同一折的 train/val/test、输入特征、归一化统计、窗口末帧和有效窗口。
- A：整数 0–7，8 类 CE，background 是第 8 类。
- B：整数 0–6，7 类 CE，加独立 background head；真实 background 不进入 step CE。
- C：7 路 soft peak BCE，加与 B 相同的独立 background head。
- 不额外新增 logits 输出项目，不增加后处理变量；现有分类头自然输出 logits。
- 本轮结论是对整套监督设计的比较。A→B 同时涉及输出结构、mask 和 loss；B→C 同时涉及目标表示和 CE/BCE，不宣称是单独某个数学函数的纯因果效果。

## 2. 已有 claude_l1 数据的实际核查

数据位置：

`G:/.shortcut-targets-by-id/1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC/ITECH_Thesis/Videos/dataset/skeleton_3d/ceiling_panel_installation_04/claude_l1`

本次只读扫描了所有 15 折，而不只是某一个样本文件：

- 15 份 manifest 的参与者划分正确：每折 12 人训练、2 人验证、1 人测试，三者互不重叠；验证人选符合已有 LOSO 规则。
- 15 份 norm 文件的 SHA-256 均匹配对应 manifest，均值/std 有限且 std > 0；全部特征共 251 维。
- 15 折拥有 15 个不同的 normalization_id。
- 共检查 **2,520 份 NPZ**：文件存在、必需字段存在、normalization_id 和 feature_profile 与所在折一致、UID 属于对应 split。
- 增强文件只位于 train，都是 `_aug-01`；每折 train 文件中一半是原件、一半是镜像。val/test 无增强。
- 各折 test 合计覆盖 **93 份不同原始 recording**。对这些记录逐一读取标签，检查整数取值及向量形状、peak 的有限性及 [0,1] 范围，未发现异常。
- 上述检查未重新由所有原始 PT 计算 norm，也未逐元素重算全部特征归一化；它确认导出结构、身份与保存统计的一致性。

### 各折文件数量

| Fold | Train | Val | Test | Train 镜像 |
| --- | ---: | ---: | ---: | ---: |
| 01 | 156 | 12 | 3 | 78 |
| 02 | 144 | 12 | 9 | 72 |
| 03 | 150 | 12 | 6 | 75 |
| 04 | 150 | 12 | 6 | 75 |
| 05 | 156 | 9 | 6 | 78 |
| 06 | 144 | 15 | 6 | 72 |
| 07 | 144 | 12 | 9 | 72 |
| 08 | 150 | 12 | 6 | 75 |
| 09 | 150 | 12 | 6 | 75 |
| 10 | 156 | 9 | 6 | 78 |
| 11 | 150 | 12 | 6 | 75 |
| 12 | 150 | 12 | 6 | 75 |
| 13 | 150 | 12 | 6 | 75 |
| 14 | 150 | 12 | 6 | 75 |
| 15 | 150 | 12 | 6 | 75 |

### 可用字段

| 字段 | 已有内容 | 三组实验中的用途 |
| --- | --- | --- |
| 各独立特征数组 | 16 组、共 251 列、逐帧保存 | 可以选择原方案的 7 组 91 维特征，无需重新导出 |
| `task_id` | 每帧整数 0–7 | A/B 的硬标签和统一评估真值候选 |
| `task_id_vector` | 每帧 8 列 peak | C 使用前 7 列 |
| `task_id_plateau_vector` | 每帧 8 列 plateau | 可用于标签诊断；不能未经说明就替换硬标签真值 |
| `task_progress` | 标量 progress，已除以 100 | 保持原方案的标量 progress 辅助任务 |
| `task_progress_vector` | 8 路 progress，已除以 100 | 本轮不必使用，避免同时改变辅助任务 |
| `mistake` | 每帧二分类标签 | 保留原方案 mistake 任务 |
| `valid_frame` | 原始帧位置上的有效标记 | 三组统一筛选可用时间窗口 |
| `normalization_id`、`feature_profile` | 导出身份信息 | 防止误读其他折的数据 |

类别顺序为 Pull Cables、Lift、Place、Align、Screw、Connect Cables、Clamp Coupling、background。不要套用更早旧数据中 Place 位于最后一列的顺序。

**结论：可以复用这些文件及每折各自的 norm，不必重新划分或导出。** 这次只改变读取时的窗口内采样与训练目标，不需要为 A/B/C 各生成一套 dataset。

## 3. norm 复用的边界

- 同一折 A/B/C 共用该折 NPZ 和 norm；不得用 fold_01 的 norm 配其他折 NPZ，不得把全部折的 NPZ 合在一起训练。
- NPZ 已经归一化，Dataset 不能再次归一化。norm 留作原始输入转换、身份核验和部署使用。
- 各列是独立标准化的，因此只选择其中 91 维不需要重新拟合 norm；原训练人群和统计范围保持不变。
- norm 使用训练原件及一份镜像、每 7 个原始帧采样计算。窗口内改为每 3 帧取一点不强制要求重算 norm；本轮固定该统计协议，以免多引入一个变量。
- 预处理已包含 ratios 截断到 [0,4]，azimuth 转 sin/cos。若以后比较这些变换本身，就需要回到原始 PT；不能把当前 NPZ 当作未经处理的原始数据。
- 新 baseline 应准确命名为“原 LSTM/优化设置 + 固定 LOSO 预处理与 10 fps 输入”，不是旧单次实验的逐位复现。

## 4. 已发现的标签定义差异

使用暂定的公共窗口网格：40 点、sample_interval=3、span=118 原始帧、window_hop=10，目标为窗口最后一个采样点。只根据保存的 valid_frame 筛选，93 份原始 test recording 合计 **105,415 个候选有效窗口**。

| 检查项 | 数量 |
| --- | ---: |
| `task_id == 7` 的窗口 | 24,078 |
| `task_id == 7` 与 `plateau[:,7] >= 0.5` 判断不一致 | 8,323，约占全部窗口 7.90% |
| 两种规则都判为非 background，但 `task_id` 与 `argmax(plateau[:7] + 0.001*peak[:7])` 不一致 | 1,100 |
| `task_id == 7`，但前 7 路 peak 至少有一路大于 0 | 15,388 |

这些数字是标签与网格诊断，不是模型测试成绩，也不保证与未来 loader 的最终窗口数完全相同：本次未额外逐元素扫描所有输入特征的有限性。最后一项的“大于 0”可能只是边界上的很小软响应，不能直接解释为 15,388 个标错样本。

**必须统一一个公共硬真值。** 不能让 A 使用 task_id，而 B/C 用 plateau 推导 background 或另一种 step argmax，否则比较同时改变了标签语义。

建议（待确定）：本轮以保存的 `task_id` 作为 A/B 训练硬标签及 A/B/C 的公共评估真值，background 对所有组均为 `task_id == 7`；C 额外使用 peak 前 7 列作为 step 训练目标。这样更贴近原来的整数标签 baseline。另一种有效选择是为所有组统一重建 plateau-based 硬标签，但必须三组同时更换并明确记录。

## 5. 编码前仍需确定的事项

### 5.1 C 的 background 窗口是否参与 step BCE

B 明确只在非 background 窗口计算 step CE。Claude 的既有 C 则对全部窗口计算 step BCE，独立 background head 同时接受监督。

- 若优先隔离“非 background 上的 step 监督方式”，建议 C 也仅在同一批非 background 窗口计算 step BCE，B/C 的 step mask 一致。
- 若希望完整复现 Claude 的监督设计，则 C 保留全部窗口的原始 peak；需要将“background 上也有 step 监督”记录为 C 的一部分。
- 不应默默把 background 的原始 peak 改成全零：这会改变现有软目标。若采用，必须另作明确决定。

因此三组方案还不能仅凭“soft peak”四个字完全确定；这项应在训练前写清楚。

### 5.2 类别平衡与 loss 权重

旧 notebook 当前实际上同时包含：

- `WeightedRandomSampler`，权重 `1/sqrt(count)`，有放回抽样。
- 8 类 CE inverse-frequency 权重。
- class 7 额外乘 0.5。

这些与 Claude 的普通 shuffle + BCE pos_weight cap=2 不是同一种策略。不能随 A/B/C 一起全部切换，却把结果只归因于监督方式。

建议（待确定）：第一轮用三组相同的普通 shuffle，不叠加类别重加权；保留原主干与 optimizer 等设置，但明确这是简化的 baseline。若希望尽量保留旧 sampler，也可三组共用同一公共硬标签计算的抽样权重、相同抽样序列；不要每组按自身 target 重新定义采样。

还需固定：A 的 class7 factor 是否保留；A/B 的 CE class weights；C 的 pos_weight；B/C 的 background loss 系数。建议 B/C 的 background 系数先相同，例如 0.2，不在本轮分别调优。原来的 class7 CE factor 与独立 background BCE 系数没有直接一一对应关系。

CE 与逐通道取平均的 BCE 数值尺度不同。即使外部 loss 系数相同，也不代表它们对共享主干施加相同梯度强度；应记录各项 loss，并把结果解释为固定权重下的监督设计比较，不把它夸大为 soft target 的普适优势。

### 5.3 原模型设置与辅助任务

从现存 `LSTM_train_mirror.ipynb` / `LSTM_model_train.py` 核对到以下设置，建议本轮以此作为固定候选，而不是继续沿用当前 L1 的 hidden128 / AdamW 配置：

| 项目 | 原方案配置候选 |
| --- | --- |
| 输入特征 | distance_from_center、position_x/y/z_relative_to_pelvis、joint_angles、polar_elevation、ratios，共 91 维 |
| LSTM | 单层、hidden 64、因果、取最后 hidden state |
| 共享层 | Linear → ReLU → dropout 0.2 |
| Progress | 1 个线性输出，标量 task_progress，全窗口 MSE |
| Mistake | 2 个 logits，CE |
| 辅助 loss 系数 | step/progress/mistake = 1/1/1，B/C 额外 background 项 |
| 优化器 | Adam，lr=1e-4，weight_decay=0.01 |
| Batch / 训练预算 | batch 32，60 epochs；无 scheduler，原循环无梯度裁剪 |

progress 和 mistake 必须三组保持相同；C 不能因为用了 soft peak 就顺便改为七路 progress 或单 logit mistake。本轮不增加后处理规则。

batch32、60轮的45次训练会比此前 batch256、12轮的 L1 明显增加计算量；是否采用完整旧预算需在运行前确定，不能跑到一半只缩短某一组。

### 5.4 时间网格

已确定窗口内 sample_interval=3，40 点。仍需正式固定 window_hop：建议沿用 Claude 的 **10 个原始帧**，约每 0.333 秒产生一个训练窗口；这与窗口内 10 fps 的采样间隔是两个参数。

如果希望测试每 0.1 秒一次的输出，则可将测试末帧网格设为每 3 帧；需要三组统一，并明确它与 Claude hop10 评估不同。

整段 118 帧内的有效性筛选应在三组保持一致，包括未被采到的中间帧。后续 feature ablation 也最好冻结这批窗口，避免删除某些特征后额外纳入不同样本。未来加入速度/加速度时，需注意抽帧不会重算这些运动特征，实时特征计算的时间单位应一致。

### 5.5 选模与报告

建议统一按验证集 **完整 8 类 macro-F1** 选择 checkpoint，并另外保存最低 val loss checkpoint。B/C 用相同固定 background 阈值（建议 sigmoid >0.5）输出 class7，其余取 step argmax。阈值如果调整只能用验证集。

完整8类混淆必须包括真实 background 样本；不能只画排除 idle 的图来证明背景问题改善。

同时报告：

- 15 折完整8类 macro-F1、accuracy 的均值与标准差。
- 逐类 precision/recall/F1、background F1。
- 15 折原始测试计数相加再按行归一化的8类 confusion matrix。
- 每名参与者的逐类 recall、A→B 和 B→C 配对变化。
- 可选7类诊断矩阵：必须另行说明 A 的 background 预测如何处理。不能通过直接丢弃 A 的 background 列而悄悄把漏检排除；若统一采用只在7个 task logits内 argmax，则三组均采用这一条件诊断规则，并与完整8类结果分开。

同折训练主干、共享层、公共辅助头应尽量复用相同初始化，采样使用独立 RNG；只设置同一个 seed 不保证不同输出头构造顺序下的公共权重完全一致。先一折短跑排查，再完成45次正式训练；主要结论后续可补不同随机种子。

## 6. 当前代码能否直接用于本实验

不能原样启动现有 L1 runner：

- `LSTM_databuilder_tune.py` 的 stride 是 window hop，当前只切连续帧，不支持窗口内 interval3。
- 当前 loader 固定返回 peak、7路 progress 和 plateau-based idle；虽然 NPZ 含 scalar task_progress，loader 并未把它作为标量辅助目标返回。
- 当前模型固定7路 step / 7路 progress / binary mistake / idle，和本轮拟保留的标量 progress、2类 mistake 不同。
- 当前 `LSTM_engine_tune.py` / runner 按非 idle 的7类 val F1选模；本轮建议按完整8类 val F1选模。

这些是之后实现独立 A/B/C 入口时要处理的地方。已有数据导出和每折 norm 不需要因此重做；本次没有改动上述任何代码。

## 7. 当前目录用途

此目录专用于本轮 step 监督方式消融。现在只有检查文档；待上述协议确定后，再加入训练入口、配置及报告工具。现有 LOSO、部署训练、notebook 和结果均保持原样。
