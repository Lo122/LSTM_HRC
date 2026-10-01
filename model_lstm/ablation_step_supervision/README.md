# Step 监督方式 A/B/C：15 折 LOSO

三种模型在 `models.py` 分为三个类；共用 `data.py`、`engine.py`、`run_loso.py` 和 `report.py`。只读取已有 claude_l1，不重新导出或修改数据，不影响旧 notebook、L1 LOSO 和部署训练入口。

## 已确认的协议

| 项目 | 设置 |
| --- | --- |
| A / `AIntegerCE` | 整数 task_id，8 类 CE，含 background |
| B / `BSeparatedCE` | 7 类 CE + 独立 background BCE |
| C / `CSoftPeak` | 7 路 soft peak BCE + 独立 background BCE |
| 真值 | 三组均为 NPZ 中 `task_id`；background = task_id == 7，不使用 plateau 重建真值 |
| B/C step mask | 仅 task_id != 7；按参与计算的窗口平均，C 还对7个通道平均；全 background batch 的 step loss 为可微零 |
| Background loss | B/C 全部窗口计算，无 pos_weight，系数均为 0.2 |
| 类别平衡 | 普通 shuffle、无放回、无类别/样本权重，无 class7 降权，无 BCE pos_weight |
| 辅助任务 | 三组均为标量 progress MSE、两类 mistake CE，覆盖全部窗口；step/progress/mistake 系数为 1/1/1 |
| 特征 | 原方案7组91维，列顺序见下文 |
| LSTM | 单层 hidden64；共享 Linear + ReLU + dropout0.2；末帧输出 |
| 时间 | 原始30fps，每隔3帧取一点，共40点，覆盖118帧位置（首末相隔3.9秒）；窗口移动10个原始帧 |
| 优化 | Adam，lr=1e-4，weight_decay=0.01，batch32，12 epochs；无 scheduler、无梯度裁剪 |
| LOSO | 15 折；沿用各折12人训练、2人验证、1人测试及各自 norm |
| 主 checkpoint | 完整8类 validation macro-F1 最大；显式包含0..7，分数相同时保留更早的 epoch |
| 额外 checkpoint | 同时保存总 val loss 最低的权重，默认测试不用它 |
| B/C 最终判断 | sigmoid(background_logit) > 0.5 输出7；否则取7个step logits的argmax |
| A 最终判断 | 8个logits的argmax |
| 随机性 | 相同折使用相同公共模块初始化、shuffle种子和独立dropout种子；B/C初始参数完全相同。CUDA不承诺跨设备逐位一致 |

固定任务间系数不属于类别加权。CE 与 BCE 的尺度本身不同，本实验比较固定配置下的监督设计，不声称将软标签和损失函数的贡献单独拆开。

特征顺序：distance_from_center、position_x_relative_to_pelvis、position_y_relative_to_pelvis、position_z_relative_to_pelvis、joint_angles、polar_elevation、ratios。数据已经过 L1 的预处理和标准化，不能再次 normalize；progress 已为 0–1，不能再次除以100。

## 运行

在仓库根目录、使用安装了 PyTorch / numpy / pandas / sklearn / matplotlib 的 Python 环境：

```powershell
python -B -m model_lstm.ablation_step_supervision.run_loso
```

默认跑 A/B/C × 15 折，每折12轮，共45次训练。按 fold 顺序加载一次数据，然后依次训练 A/B/C，三组复用同一批窗口。默认源目录已设为 G 盘 `ceiling_panel_installation_04/claude_l1`。

单折短跑（用于检查，不是正式15折成绩）：

```powershell
python -B -m model_lstm.ablation_step_supervision.run_loso --folds 1 --epochs 1
```

只运行一个方案，仍默认为15折、12轮：

```powershell
python -B -m model_lstm.ablation_step_supervision.run_loso --arms B
```

也可在 IDE 直接运行 `run_loso.py`。`--help` 列出 data-root、output-dir、device、seed 和 cache-dir 等选项。默认自动选择 CUDA（可用时），否则 CPU。

中断后指定原来的实验目录即可继续（替换下面的目录名）：

```powershell
python -B -m model_lstm.ablation_step_supervision.run_loso --output-dir "model_lstm/ablation_step_supervision/runs/ABC_原来的时间戳"
```

自动跳过已有 `test_results.json` 的组别/折；未完成的组别/折从第1轮重新训练，不恢复中途epoch，旧训练日志会重写。可用 `--folds 3 4 5 6 7 8 9 10 11 12 13 14 15` 限定本次运行范围，或用 `--arms B C` 只补跑部分方案。汇总始终包括目录中 A/B/C 所有已完成的折。已全部完成时，该命令只重新汇总，无需加载训练数据。

续跑须保持原实验的 epochs、seed、数据路径和训练设置一致（例如烟雾测试需继续指定 `--epochs 1`）；脚本也会核对已保存的数据清单和norm。原始代码快照保留。未指定 `--output-dir` 仍创建新实验。不要让两个进程同时写入同一个实验目录。

## 输出

默认路径：`runs/ABC_<时间戳>/`。此目录已在本文件夹的 .gitignore 中排除，不把训练产物当作代码提交。

```text
ABC_<时间戳>/
  protocol.json                 实际配置，包括运行的 arms/folds/epochs
  source/                       本次实验代码和说明快照
  comparison.csv                各组完整8类F1均值、标准差、accuracy和background F1
  paired_differences.csv         同一参与者的 A→B、B→C、A→C F1变化
  A/                            B/、C/同结构
    fold_01/ ... fold_15/
      config.json
      norm_stats.npz             原折norm的副本，与该折NPZ匹配
      dataset_manifest.json
      metrics.csv               每轮train/val指标
      best_macro_f1.pth/json     主checkpoint及对应验证指标
      best_val_loss.pth/json     最低总val loss checkpoint
      test_predictions.csv
      test_results.json
      confusion_matrix.csv/png
      confusion_matrix_norm.csv/png
      per_class.csv
    fold_results.csv
    loso_summary.json            测试参与者等权均值/std（ddof=0）
    participant_recall.csv
    confusion_matrix.csv/png    所有已完成折的测试计数合并
    confusion_matrix_norm.csv/png
    per_class.csv               汇总计数得到的逐类指标
```

启动时以及每完成一个组别/折后更新汇总；`test_results.json` 在预测和报告全部导出后原子写入，作为完成标记。各组汇总只包括实际完成的折。

## 评估含义

- 主指标 `macro_f1_8` / `accuracy_8` 包括真实 background、预测 background 和所有对应错误；checkpoint 仅由验证集选择。
- `background_f1` 由完整8类最终预测计算，A/B/C口径相同。
- 辅助 `task_macro_f1_7_conditional` 在真实 task_id != 7 的窗口上，三组都只对前7个step logits做argmax，完全忽略background决策。这是条件诊断，不能替代8类主指标。
- `test_predictions.csv` 的 `true_step` 和 `pred_step` 均为0–7（与旧 L1 CSV 的 true_step=-1 排除标记不同）；`pred_task_conditional` 仅用于上述7类诊断。
- CSV 另含 test_uid、file、target_frame（零起始原始帧位置），可验证三组逐窗口对应。
- `metrics.csv` 的 train 指标来自训练过程中含 dropout 的预测，不是固定checkpoint对训练集完整重评估；val/test 使用 eval 模式。
- pooled confusion matrix 是原始计数相加后按行归一化，参与者样本数隐式影响各行，不等于15张归一化矩阵的平均。`loso_summary.json` 才是参与者等权平均。
- 使用少于15折的 smoke run 时，汇总只标注实际完成折数，不会冒充完整LOSO。

实施前完整数据核查及标签定义差异保留于 `PRE_IMPLEMENTATION_AUDIT.md`。

## 验证

```powershell
python -B -m unittest model_lstm.ablation_step_supervision.test_ablation -v
```

测试覆盖时间采样与末帧、无效帧筛选、统一硬真值、B/C背景mask及梯度、无权重loss、公共初始化、8类指标、验证选模，以及合成数据上的完整 A/B/C 单折训练/预测/汇总。不会启动真实数据的45次训练。
