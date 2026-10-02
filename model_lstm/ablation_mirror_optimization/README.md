# Initial mirror: L1 optimization settings ablation

以已完成的 `loso_initial_mirror` 为对照，只替换优化配置。模型、数据、loss、采样和评估代码继续复用原实现。

| 设置 | 原对照 | 本实验 |
|---|---|---|
| optimizer | Adam | AdamW |
| learning rate | 1e-4 | 1e-3 |
| weight decay | 0.01 | 0.0001 |
| scheduler | 无 | CosineAnnealingLR，T_max=epochs（默认12），eta_min=0 |
| gradient clipping | 无 | loss.backward后、optimizer.step前，global norm最大1.0 |

固定89维特征、160帧窗口、stride30、hidden64、dropout0.2、batch32、12epochs；保留平方根逆频率replacement sampler、平方根逆频率CE及weighted_sample_mean、class7 factor1、progress/mistake及全部辅助loss系数。保留fold seed=42+uid-1及相同初始化、采样顺序。梯度裁剪是此次优化配置的一部分；并未把L1的batch256、特征、监督或其他训练设置带入。

默认对照为 `model_lstm/loso_initial_mirror/runs/MIRROR_LOSO_2026-10-01_15-38-16_506620`。正式运行前核对对照的参数、所选折完成情况和norm哈希。每折原样复制其norm和数据清单，再验证原始源文件，复用本地raw缓存；不重拟合norm，不读入对照模型的训练权重。模型从同种子重新初始化。

## 运行

```powershell
python -B -m model_lstm.ablation_mirror_optimization.run_loso
```

默认15折、每折12轮；结果写入本目录 `runs/MIRROR_L1OPT_<时间戳>/`，不覆盖对照。可以指定其他兼容的 `--baseline-run`。与原入口相同，支持 `--folds 1 2`、`--device cpu`、`--cache-dir`；epochs、seed、selection必须与对照相同，不能缩短训练预算混入比较。小数据测试使用独立合成对照，不用正式15折作烟雾测试。

中断后使用原输出目录：

```powershell
python -B -m model_lstm.ablation_mirror_optimization.run_loso --output-dir "model_lstm/ablation_mirror_optimization/runs/MIRROR_L1OPT_原来的时间戳"
```

跳过已完成折，未完成折从epoch1重跑；不在epoch内续训。不同时向同一目录运行两个进程。

## 输出

与initial mirror相同，mirror/fold_XX保存双checkpoint、norm、manifest、训练曲线CSV、train/val/test结果、test predictions以及confusion matrix。默认由validation完整8类Macro F1选checkpoint，test不参与选择。train评估完整遍历，无weighted sampler。

每折结束自动重新汇总：mirror/loso_summary.json、participant_recall.csv、pooled confusion matrix等。

额外输出根目录：
- paired_optimizer_comparison.csv：相同参与者上，新配置减对照的F1、accuracy等差值。
- optimizer_comparison_summary.json：只在共同完成折上计算两个方案的均值/std及配对差值，明确参与者列表。

比较前校验每个test窗口的file、target_frame、true_step，以及每折norm和manifest完全相同。记录完整实现快照，并保留baseline_run来源。

本实验检验整组优化设置的影响，不能单独据此认定AdamW、学习率、正则化或裁剪中的哪一项贡献最大。

## 全数据部署训练

```powershell
python -B -m model_lstm.ablation_mirror_optimization.deploy_train
```

全部15名参与者的原始文件及每份的一次aug-01镜像参与训练。沿用本轮优化设置、89维特征、160帧、batch32、sampler和加权loss；重新拟合全数据norm，重新计算全部训练窗口的类别权重。默认固定12轮，与成功的LOSO预算一致，不使用测试分数选轮数或模型。输出本目录 `runs/MIRROR_DEPLOY_<时间戳>/`，不覆盖LOSO。

保存最终model_weights.pth、norm_stats.npz、config.json、数据清单、训练记录和独立于仓库的inference.py/model_definition.py。全数据训练无validation，不存在best validation checkpoint；最终模型是第12轮。训练期间每轮更新last_model_weights.pth和last_checkpoint.json，正常完成后自动删除；中断时保留last权重，但不支持恢复optimizer/scheduler的续训。已有输出目录禁止覆盖；epochs可显式覆盖，配置记录为非默认预算。

推理使用160帧89维**未归一化**特征，按config中的feature_layout排列。DeploymentModel自动加载与权重配套的norm，返回8类step logits/softmax probabilities（含背景7）、step id、progress及mistake预测。不要使用某个LOSO折的norm，不要二次归一化。部署训练结束后的train_results与confusion matrix是训练集诊断，不是泛化指标。

## 验证

```powershell
python -B -m unittest model_lstm.ablation_mirror_optimization.test_optimization -v
```
