# Claude LSTM / GRU 复现评审

最新决定（2026-09-29）：用户要求完整对齐 Claude L1（LSTM）的设置。已更新 `_tune` 代码并添加 15 折入口；18 项测试通过，尚未运行完整训练。当前配置、修改清单与运行方式见 [LSTM_plan_tune.md](../model_lstm/LSTM_plan_tune.md)；下文保留初始评审，不代表当前代码状态。

评审日期：2026-09-28。参考文件：`C:/Users/loy49/Desktop/LSTM_GRU_Model_Settings.md`。

结论：文档足以确定主要实验协议，本地具备初步复现条件。但这涉及输入、标签、输出头、损失和评估协议的整体对齐，不能只修改现有 notebook 的 hidden size、learning rate 等参数。文档报告的 S3 指标尚未在本仓库复现；本次完成代码对照、数据抽查与环境核验，没有启动训练。

## 1. 已核验的条件

- 本地数据根目录：`G:/.shortcut-targets-by-id/1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC/ITECH_Thesis/Videos/dataset/skeleton_3d/ceiling_panel_installation_04`。
- `original/` 有 93 个 `.pt`，UID 为 1–15，与文档的数量一致。这不证明内容与 Claude 的 `data_f30a14d` 快照一致。
- 当前 `augmented_mirror/` 根目录没有 `.pt`；镜像文件在 `aug_train/raw`（225）、`aug_val/raw`（27）、`aug_test/raw`（27）。旧划分分别为 UID 3–14、1/15、2。重做 LOSO 时，目录名不能作为新的数据划分依据。
- 抽查 `features__cam-05_uid-01_take-02.pt` 及其 `_aug-01`：均为 17,571 帧，原始元数据为 30 fps；包含 16 组特征、8 列 peak/plateau/progress 标签；两份文件的全部标签张量逐元素一致，特征均有限。尚未逐文件验证全量数据。
- 抽查的特征列顺序符合文档；原始特征为 235 维，azimuth 由 16 维角度变为 32 维 sin/cos 后为 251 维。
- 文档中 15 组 validation UID 已用 `np.random.RandomState(1000 + k)` 重算，全部一致。
- `hrc_communication/.venv/Scripts/python.exe` 实测：Python 3.10.20、PyTorch 2.11.0+cu128、NumPy 1.26.4、sklearn 1.7.2，CUDA 可用。文档 Python 为 3.10.19，其余上述软件版本一致。
- Git 历史中存在 `f30a14d` 和对应 `dataset/labels.py`；当前工作树已无该标签生成实现。未找到文档所述 `bench/loso.py` 等源代码和逐折结果。

## 2. 当前实现与目标实验的区别

“当前”指工作树中的 `model_lstm/LSTM_train_mirror.ipynb` 和 `LSTM_model_train.py`，并非所有历史 checkpoint 的训练实现。

| 项目 | 当前实现 | Claude S3 复现目标 |
| --- | --- | --- |
| 输入 | 7 组特征、91 维 | 全部 16 组特征、251 维 |
| 角度/ratio 处理 | 当前选择不含 azimuth；ratios 直接归一化 | azimuth 转 radians 后拼接 sin、cos；ratios clip 到 [0,4] |
| 归一化 | 旧训练集的固定统计文件 | 每折只用该折训练原件及一份镜像；每 7 帧采样 |
| 时间窗口 | 连续 160 帧，窗口移动 30 帧 | 80 个采样点，点间 3 个原始帧，窗口移动 10 个原始帧 |
| 主干 | 1 层 LSTM，hidden 64，dropout 0.2 | 1 层 GRU，hidden 128，共享层 dropout 0.3 |
| 任务头 | 8 类，包括 idle；标量 `task_id` | 7 个独立 logits；用 peak 向量训练 |
| 进度头 | 1 个标量 | 7 路输出；仅 plateau >= 0.5 的通道参与 MSE |
| mistake 头 | 2 个 logits，CE | 1 个 logit，加权 BCE |
| idle 头 | 无独立头 | 1 个 logit，BCE，权重 0.5 |
| 任务损失 | inverse-frequency 加权 CE，idle 额外乘 0.5 | BCEWithLogits；pos_weight = neg/pos，上限 2 |
| 采样 | WeightedRandomSampler，当前代码为 1/sqrt(count) | 普通 shuffle，无重采样 |
| 优化器 | Adam，lr 1e-4，weight decay 0.01 | AdamW，lr 1e-3，weight decay 1e-4 |
| 训练 | batch 32，60 epochs；无 scheduler/梯度裁剪 | batch 256，10 epochs；CosineAnnealingLR；梯度范数上限 1 |
| loss 系数 | step/progress/mistake = 1/1/1 | task/progress/idle/mistake = 1/0.3/0.5/0.3 |
| 划分 | 固定 train/val/test，测试 UID 2 | 15 折 LOSO，每折 12 人训练、2 人验证、1 人测试 |
| 评估数据 | 旧划分中的增强文件 | val/test 只使用原始文件 |
| 最佳模型选择 | 当前代码已额外保存并使用 macro-F1 checkpoint，按 8 类计算 | 按真实非 idle 窗口的 7 类 validation macro-F1 |

现有 `data_proc_2d/app/build_norm_dataset.py` 导出的 `.npz` 仅保存标量 `task_id`、`task_progress`、`mistake`。无法从这些标量恢复重叠任务的 soft peak、plateau 或逐类 progress；复现必须读原始 `.pt` 或重新导出完整向量。

当前 notebook 的 sampler 实际已使用 `1/sqrt(count)`，但日志、配置中的 `inverse_window_class_frequency` 等描述仍写成逆频率，并打印均衡采样占比。这些记录不能替代对代码的核查。复现方案不使用 sampler，因此无需把这套逻辑带入。

## 3. 必须按原定义实现的细节

1. **两种 stride 的含义不同。** 当前 Dataset 的 `stride` 是窗口 hop。Claude 的 `stride` 是窗口内部采样间隔；S3 应为 `x[start:start+238:3]`，共 80 个点，标签取 `start+237`，下一窗口起点加 10。有限性检查覆盖全部 238 个原始帧，包括未被采样的中间帧。窗口不能跨文件/take。
2. **任务通道独立。** 输入 loss 的 peak 不做 softmax、不强制行和为 1，也不先 argmax 成单标签。idle 仍参与 task BCE，其任务目标为原始的零值或近零值。不要改用已保存的标量 `task_id`。
3. **训练目标和评估真值不同。** task loss 使用 `peak[:7]`；评估真值使用 `argmax(plateau[:7] + 1e-3 * peak[:7])`。仅 `plateau[:,7] < 0.5` 的窗口参与任务指标；不是用预测 idle 过滤。
4. **不能仅凭 scalar task_id 排除 idle。** 抽查原始文件中，scalar task_id=7 有 3,699 帧，而 plateau 第 7 列 >=0.5 只有 2,550 帧，两种筛选定义不同。
5. **标签 ID 和向量列位置不同。** 抽查原始元数据的 `task_names` 中 No Related Task 的 ID 仍为 8，实际 task 向量只有 8 列，idle 在列位置 7，标量 task_id 已是 0–7。历史 labeller 的列键为 `[0,1,2,3,4,5,6,8]`。读取向量应按已核验的列顺序，不能用元数据 ID 8 直接索引。
6. **镜像只属于训练。** 汇总旧目录中的 `_aug-01` 后，按当前折的训练 UID 选择，并与原件一一配对。val/test 不使用旧增强文件。不能加入 `_aug-02/03` 增加重复权重。仓库此前审计仅证明当时选用的 6 组特征及标量标签重复；全 16 组特征的重复性尚未在本次核验。
7. **每折重新计算归一化与 pos_weight。** 统计范围不能包含该折的 val/test；pos_weight 按训练窗口末帧 peak >=0.5 计正样本，而不是按 scalar task_id 或所有原始帧计数。
8. **progress 只在活动通道上求平均。** 分母是活动通道数量，不是 batch_size×7；评估 MAE 还排除真实 idle 窗口，并乘 100。原始 Linear 输出是否额外约束须以 Claude 源码为准，不能自行添加 sigmoid/clamp。
9. **取最佳验证 F1 的 epoch。** 当前 notebook 已有 F1 选模，但口径不同；不是简单换 checkpoint 文件名就能对齐。每折都应在恢复最佳权重后才评估 test。
10. **S3 的 stride 不表示重新计算速度特征。** 按文档，应先使用现有 30 fps 帧级特征，再隔 3 帧采样。重新以 10 fps 计算速度/加速度会改变输入。

## 4. 怎样理解报告中的性能

S3 报告的是 15 折、每折先计算的 7 类 macro-F1 的均值和标准差：`0.499 ± 0.086`。它不是 accuracy，也不是把所有人的窗口拼在一起算一次 F1，更不是模型同时判断 idle 和任务时的端到端正确率。任务指标排除了**真实** idle 窗口；idle F1 另算，因此不能据此推断 idle gating 后的整体表现。

从当前保存的 test CSV 重新计算得到：

| 历史运行 | 窗口数 | 8 类 accuracy | 8 类 macro-F1 | balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| exp_2026-09-25_15-28-47 | 7,800 | 0.4292 | 0.2983 | 0.3544 |
| exp_2026-09-25_19-34-01 | 7,800 | 0.3458 | 0.2543 | 0.3253 |

这些数值包含 idle，使用旧标签定义、单人测试和旧增强文件，不能直接与 S3 的 0.499 作增益计算。它们只帮助澄清当前保存的指标是什么。

文档中的 B0/L1 是较干净的 GRU/LSTM 对照，报告均值接近（0.465/0.456）。仅凭这些数字没有理由把主要改进归因于 GRU。B0 到 S3 同时改变采样间隔、时间范围、epoch 数和 idle loss 权重；无法区分各项贡献。更丰富的运动特征、soft 多任务标签、训练分布和更长上下文都是值得检验的因素，目前只是解释假设。

progress MAE 24.1 是 0–100 标度上的活动任务通道误差，不表示进度预测已很精确；文档也未报告 mistake 的测试指标。不能将 task F1 的提升推广为所有输出均改善。

最后“13 人训练选 epoch，再全 15 人训练 4 epochs”的模型属于部署导出。全体人员已参与训练，不能用同一批人的表现重新验证 LOSO 指标。

## 5. 精确复现仍缺什么

用户已确认目前只有这份总结。可以按文档实现协议级复现，不必等待源码；实现时应明确记录下列未指定细节的选择，且不能承诺重现小数点后三位。若以后取得 Claude 的 `bench/{loso,data,models,engine}.py`、逐折结果 JSON 和数据清单/指纹，再用于精确核对。

需要由源码/运行产物确认的具体细节：

- `data_f30a14d` 与当前 93 份原件、93 份镜像的特征和标签内容是否一致；尤其 Lift trimming、idle gap、mistake progress 处理。
- 文件枚举顺序，以及每 7 帧采样与非有限值过滤的先后顺序。
- mean/std 的累计 dtype、标准差 ddof；fold 间标准化缓存是否隔离。此前仓库审计已有 float32 统计对累加顺序敏感的记录，不能随意改变统计实现后仍称逐值复现。
- macro-F1 是否显式传 `labels=range(7)`；某折缺失类别时 sklearn 的默认行为会影响均值。15 折标准差使用 ddof=0 还是 1。
- progress head 是否有输出变换；空活动 mask 的处理；mistake 无正样本时 pos_weight 的处理。
- 文件/窗口 shuffle 使用哪个随机数生成器、seed 设置时机、DataLoader worker/drop_last，以及 AMP/TF32/cuDNN deterministic 设置。
- `0.499` 的逐折值、各折最佳 epoch 和日志。文档中的“误差约 0.005”尚无本地重复实验支持，不应当作已验证的容差承诺。

## 6. 建议的复现执行顺序

1. **冻结数据协议。** 建立原件和 `_aug-01` 配对清单，保存指纹；检查所有文件的 fps、列顺序、标签形状与有限性；锁定 15 折 UID 和软件版本。
2. **独立复现实验入口。** 直接读取完整 `.pt`，实现 251 维转换、每折统计和四个输出头。配置分别保存 B0/L1/S3；避免复用只有标量标签的旧 NPZ。
3. **先运行一个固定 fold 的 S3。** 例如 test UID 2、val UID 5/14、其余 12 人训练，seed=1（fold_index，非 test UID）。对照 Claude 同折文件数、窗口数、类别权重、最佳 epoch 与指标；单折只验证流程，不用于声称复现整体结果。
4. **完成 S3 全部 15 折。** 每折保存配置、统计量、最佳 checkpoint、逐 epoch 验证指标、test 预测及混淆矩阵；报告逐折 mean/std 和 pooled recall，分别汇总。
5. **再做 B0/L1 对照。** 用同一实现仅切换文档规定的参数。若要解释改进来源，再依次消融输入特征、标签/loss、窗口、idle 权重；不把新增调参混入复现结果。
6. **最后导出部署模型。** 按文档的 2/7 验证流程选 epoch，再全 15 人重训；保存该模型自己的归一化统计，单独标明它不是 LOSO 测试模型。

本次只新增这份评审文档；未修改训练代码、数据或历史实验，也未运行 15 折训练。全文区分了本地实测、抽查、历史审计和 Claude 文档中的未验证报告值。
