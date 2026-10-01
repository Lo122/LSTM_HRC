# L1 backbone: step supervision B vs C

本实验保留 L1 的特征、网络与优化设置，对比两种 step 监督组合：

| 方案 | Step target / loss |
|---|---|
| B | 整数 task_id，七类无权重 CE |
| C | 七通道 soft peak，原 L1 的加权 BCE（pos_weight cap=2） |

两组共用独立 background head、task_id != 7 的 step loss mask、task_id == 7 的背景真值。背景窗口仍参与辅助任务。统一使用整数 task_id 作为完整8类评估真值，并按完整8类 validation macro-F1 选 checkpoint。

**C 是统一协议后的 L1-soft 对照组，不是旧 L1 的逐项复现。** 相对旧 L1，两组共同调整了 step mask、background 真值、评估真值及选模规则。因此需要重新训练 B/C，旧 LOSO 仅作参考。比较结论针对标签与 loss 组合（包括 BCE 正例权重），不能单独归因于标签形状。

## 固定设置

- 复用已有 claude_l1/fold_01..15 的数据与每折 norm，禁止重新拟合或二次 normalization。
- 原 L1 顺序的251维特征，30 fps连续120帧（约4秒），窗口 hop=10，最后一帧标签。
- 15折：每折12名train、2名val、1名test；train包含原始及一份镜像，val/test仅原始。
- 同一个 L1 LSTM：hidden=128，1层，shared dropout=0.3；step 7、progress 7、mistake 1、background 1输出。
- AdamW，lr=0.001，weight_decay=0.0001，batch=256；CosineAnnealingLR，梯度裁剪1.0；默认12轮。
- Loss 系数 step/progress/mistake/background = 1/0.3/0.3/0.2。
- Progress 保留 L1 的 plateau[:7]>=0.5 mask 和7通道 MSE，不额外添加非背景 loss mask；progress MAE 只统计非背景激活通道。
- Mistake 保留 L1 加权 binary BCE；B/C 使用相同 train-only 权重统计。
- C 的 step pos_weight 仍按原 L1：在所有训练窗口统计 peak>=0.5 的比例，上限2；仅 loss 计算采用共同的非背景 mask。B 的 CE 不使用该权重。
- 无 weighted sampler。默认基础seed=0，各折seed=uid-1；两组初始化、shuffle及dropout随机种子相同。
- 背景 sigmoid>0.5 输出7，否则七个step logits argmax；两组相同。

## 运行

在仓库根目录运行两组15折：

```powershell
python -B -m model_lstm.ablation_l1_supervision.run_loso
```

仅运行 CE 或部分折：

```powershell
python -B -m model_lstm.ablation_l1_supervision.run_loso --arms B --folds 1 2 3
```

默认输出本目录 `runs/L1_BC_<时间戳>/`。同一目录续跑会自动跳过完成组别/折，未完成的从epoch 1重跑：

```powershell
python -B -m model_lstm.ablation_l1_supervision.run_loso --output-dir "model_lstm/ablation_l1_supervision/runs/L1_BC_原来的时间戳"
```

新增C补跑时仍使用同一output-dir即可。所有训练设置、数据路径、epochs、seed必须与原实验一致。不要同时运行两个写同一目录的进程。未指定output-dir则创建新实验。可直接在IDE运行run_loso.py；导入不启动训练。

## 输出与评估

每个 B/C/fold_XX 保存 config、norm、manifest、metrics.csv、best_macro_f1.pth/json、best_val_loss.pth/json、test_predictions.csv、test_results.json、8类原始/归一化 confusion matrix 和 per_class.csv。test使用best_macro_f1，不使用最低loss模型。

启动及每完成一个组别/折，重新汇总整个目录所有已完成结果：每组loso_summary.json、fold_results.csv、participant_recall.csv、pooled confusion matrix；根目录comparison.csv与paired_differences.csv（C减B，同一test UID配对）。参与者等权均值和std（ddof=0）与pooled计数分开报告。部分折只报告实际完成折数。

七类conditional指标只用于诊断：真实非背景窗口上的七类argmax，忽略background gate；主指标始终为完整8类macro-F1。Background F1是类别7的F1。

source/保存本次实验及其共享依赖的代码快照。已完成折以最后原子写入的test_results.json为标记；续跑保留其权重及预测。

## 检查

```powershell
python -B -m unittest model_lstm.ablation_l1_supervision.test_ablation -v
```
