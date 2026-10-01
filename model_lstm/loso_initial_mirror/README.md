# Historical mirror configuration: 15-fold LOSO

冻结用户指定的 `exp_2026-09-24_17-02-55/config.json` 设置，对早期优化配置做参与者泛化评估。不是先前 ablation A，也不是 L1；不改写原 notebook、数据文件或已有结果。

## 保留的训练设置

`reference_config.json` 保存指定历史实验的配置副本；运行时提取该文件的设置，不再读取当前 notebook。来源为用户指定的 `ITECH_Thesis/model_test/clean split/exp_2026-09-24_17-02-55`。

- 按历史配置顺序使用6组特征，共89维：distance_from_center、position_x/y/z_relative_to_pelvis、joint_angles、polar_elevation。没有ratios、方位角、速度或加速度。checkpoint输入权重形状[256,89]也已核实。
- 30 fps 连续160帧（约5.33秒），窗口 stride=30，predict_offset=0，seq2one末帧标签。
- 原 AssistLSTM：hidden=64、1层、shared dropout=0.2，step8类、progress标量、mistake2类。
- 60 epochs，batch=32；Adam lr=1e-4、weight_decay=0.01；无scheduler、无gradient clipping。
- WeightedRandomSampler：每个窗口权重 `1/sqrt(class_count)`，replacement=True，每轮样本数等于全部训练窗口数。**历史config的sampler_weighting写inverse_window_class_frequency，但test_notes明确写sqrt balanced sampler，此前notebook代码也有同类元数据不一致。本实验按备注使用平方根逆频率；仅凭该config无法证明历史sampler的实际代码。**
- CE类权重 `1/sqrt(count)`，以原始训练分布加权均值归一为1，class7不额外衰减（factor=1）。历史counts可复现全部保存权重，最大绝对误差约1.2e-7。每折重新计算权重，不直接复用历史数值。使用 `reduction='none'` 后对样本取mean，不能替换为PyTorch weighted CE的默认mean。
- Step weighted CE、progress MSE、mistake unweighted CE，系数1/1/1。progress由原始百分数除100一次。

## LOSO与归一化

与已有L1/ABC相同的15组参与者划分：每折12 train、2 val（RandomState(1000+test_uid)）、1 test。训练原始文件＋每个文件的一份aug-01镜像，val/test只有原始文件。旧单次划分不复用。

每折从原始PT计算本折train-only norm，包括训练镜像。保持旧算法：全部训练帧、float32逐特征维度population mean/std，std加1e-6，normalize时std<1e-6改为1。无额外特征变换、无每7帧采样。按特征组拼接计算，降低峰值内存。

原始特征只在本地`.dataset_cache/initial_mirror_raw/`缓存一次，各折逐窗口使用自己的norm，不生成15份大型normalized数据集。该缓存不是临时Python字节码，不应删掉。每折单独保存norm_stats.npz、源文件清单、UID划分和norm哈希。源PT和旧norm不修改。task_id 8只在内存映射为7，与原始预处理规则一致。

遇到非有限原始特征直接报错，不静默删除帧或插值，以免改变历史采样。新增了可重复的fold seed=42+uid-1；原notebook只明确指定了sampler seed，没有固定模型初始化，因此这是配置复现，不是历史随机轨迹的逐比特重放。

## 运行

```powershell
python -B -m model_lstm.loso_initial_mirror.run_loso
```

默认15折，每折60轮。首次缓存与计算统计量需要时间。先检查一折一轮可用 `--folds 1 --epochs 1`（单独实验，不能与60轮结果混合）。

支持 `--data-root`（ceiling_panel_installation_04，内含original与镜像来源目录）、`--folds`、`--epochs`、`--seed`、`--device`、`--cache-dir`。直接在IDE运行也支持。

默认使用**验证集完整8类macro-F1最高**的checkpoint测试。两个checkpoint都保存：best_macro_f1与best_val_loss。历史实验保存了两个checkpoint，但config未注明当时test加载了哪个。默认macro-F1用于统一LOSO比较；如选择最低val loss，启动独立实验时指定：

```powershell
python -B -m model_lstm.loso_initial_mirror.run_loso --selection val_loss
```

不能查看test结果后挑选规则。首次运行前确定selection，续跑禁止改变。

## 输出与中断

输出 `runs/MIRROR_LOSO_<时间戳>/mirror/fold_01..15/`：配置、norm、源manifest、双checkpoint、metrics.csv、train/val/test_results.json、test_predictions.csv及8类confusion matrix。选定checkpoint对train完整无sampler遍历；train/val矩阵另存train_evaluation/、val_evaluation/。

启动及每完成一折重新生成mirror/loso_summary.json、fold_results.csv、participant_recall.csv、pooled confusion matrix、根目录comparison.csv。summary均值/std为参与者等权，pooled矩阵累加样本计数；只报告实际完成折数。test_predictions含文件名和原始末帧位置。

```powershell
python -B -m model_lstm.loso_initial_mirror.run_loso --output-dir "model_lstm/loso_initial_mirror/runs/MIRROR_LOSO_原来的时间戳"
```

同目录自动跳过完成折，未完成折从epoch1重新训练，复用已验证norm与缓存；test_results.json是原子写入的完成标记。不恢复epoch中间状态。保持相同epochs/seed/selection，不要同时两个进程写同目录。历史config及依赖代码保存在source快照。修正前的235维实验不能续用，协议版本已更新，必须新建运行。

## 比较范围

这是早期优化配置的LOSO结果，不是最早未经调参的初始版本。训练60轮与ABC/L1的12轮不同，窗口长度和评估末帧也不同；不能把跨方案分数差异仅归因于step监督方式。历史调参使用过的参与者信息无法通过回顾性LOSO消除，应在论文中说明。

## 检查

```powershell
python -B -m unittest model_lstm.loso_initial_mirror.test_loso -v
```
