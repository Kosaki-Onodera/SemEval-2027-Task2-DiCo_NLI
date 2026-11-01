# DiCo-NLI 实验与系统描述记录

> 用途：为 SemEval 2027 Task 2 English track 的系统描述论文保存可追溯的实验事实。
>
> 最后更新：2026-11-01
>
> 范围：数据、代码变更、训练配置、每轮结果、异常、修复、产物和待验证结论。

## 1. 记录原则

本文件只记录对论文、复现或后续决策有影响的信息，不记录普通对话细节。

每次新增实验时，应至少补充：

1. 实验日期和实验编号；
2. 使用的数据 split；
3. 基座模型和训练方式；
4. 所有影响结果的超参数；
5. 每个 epoch 的 Weighted F1、SoftCons、HardCons；
6. 遇到的环境或训练问题；
7. 最终使用的 checkpoint/adapter；
8. 与已有实验的严格对比；
9. 该结果是否属于正式实验、smoke test 或失败运行。

## 2. 当前一句话总结

English track 上已完成两种有效系统：

- ModernBERT-base 全参数微调：当前 Weighted F1 和 SoftCons 最高；
- DeBERTa-v2-xxlarge LoRA：当前 HardCons 最高，但 `NEGATIVE_OTHER` 识别明显较弱。

当前保存的最佳 DeBERTa LoRA 是 epoch 3。该模型使用普通 LoRA，不是 QLoRA，也不是 Unsloth。

## 3. 数据

### 3.1 任务数据

| Split | English 实例数 | 用途 |
|---|---:|---|
| `final_data/train` | 3,042 | 训练 |
| `final_data/dev` | 660 | 独立开发验证、scorer 评分 |
| official test | 未发布 | 后续通过 CodaBench 提供 |

训练和验证文件：

```text
final_data/train/dico_nli_train_track1_participant_labeled.csv
final_data/dev/dico_nli_dev_track1_participant_labeled.csv
final_data/dev/dico_nli_dev_track1_reference.csv
```

标签分布：

| Split | BACKWARD | EQUIVALENCE | FORWARD | NEGATIVE | Total |
|---|---:|---:|---:|---:|---:|
| Train | 881 | 802 | 881 | 478 | 3,042 |
| Dev | 190 | 174 | 190 | 106 | 660 |

### 3.2 Trial 数据

Trial 数据共 797 个 English 实例，只用于 smoke test 和流程验证，不能作为正式模型质量结论。

## 4. 输入表示与标签

当前 starter kit 将两个短语编码为单段文本：

```text
text1 [SEP] text2
```

SEP 优先使用 tokenizer 自带的 `sep_token`。

模型输出四分类：

```text
BACKWARD_ENTAILMENT
FORWARD_ENTAILMENT
EQUIVALENCE
NEGATIVE_OTHER
```

方向反转规则：

```text
Rev(FORWARD_ENTAILMENT)  = BACKWARD_ENTAILMENT
Rev(BACKWARD_ENTAILMENT) = FORWARD_ENTAILMENT
Rev(EQUIVALENCE)         = EQUIVALENCE
```

`NEGATIVE_OTHER` 不参与 SoftCons 或 HardCons。

## 5. 实验总览

| ID | 模型 | 训练方式 | Split | 最佳 epoch | Weighted F1 | SoftCons | HardCons | 状态 |
|---|---|---|---|---:|---:|---:|---:|---|
| E1 | ModernBERT-base | Trial smoke | trial dev | 3 | 0.8372 | 0.8125 | 0.7411 | 非正式 |
| E2 | ModernBERT-base | 全参数微调 | final dev | 3 | **0.6925** | **0.7762** | 0.6606 | 正式开发实验 |
| E3a | DeBERTa-v2-xxlarge | FP16 LoRA，pooler 冻结 | final dev | 1 | 0.2719 | 0.1119 | 0.0686 | 失败 |
| E3b | DeBERTa-v2-xxlarge | FP32 LoRA，pooler 冻结 | final dev | 3 | 0.3402 | 0.2996 | 0.1661 | 失败/诊断 |
| E4 | DeBERTa-v2-xxlarge | FP32 LoRA，pooler 可训练 | final dev | 3 | 0.6802 | 0.7437 | **0.6751** | 有效开发实验 |
| E5 | DeBERTa-v2-xxlarge | FP32 LoRA，N 权重 1.5 | final dev | 3 | 0.4395 | 0.4404 | 0.3502 | 失败，已拒绝 |
| E6 | E4 adapter + dev logit 校准 | 不重新训练 | final dev | bias=0 | 0.6825 | 0.7437 | 0.6751 | 主候选 |

## 6. E1：ModernBERT trial smoke test

### 6.1 目的

验证训练、反向传播、预测 CSV、官方 scorer 和三项指标的完整链路。

### 6.2 数据与限制

- 数据：`trial_data`；
- train/dev/test 人为复用相同 trial 数据；
- 结果仅用于工程验证；
- test 分数与 dev 分数相同，不能视为泛化性能。

### 6.3 结果

| Epoch | Weighted F1 | SoftCons | HardCons |
|---:|---:|---:|---:|
| 1 | 0.5617 | 0.3810 | 0.3006 |
| 2 | 0.7683 | 0.7202 | 0.6339 |
| 3 | 0.8372 | 0.8125 | 0.7411 |

论文中不应报告该组数字为正式 benchmark 结果。

## 7. E2：ModernBERT-base 全参数微调

### 7.1 配置

| 项目 | 值 |
|---|---|
| 模型 | `ModernBERT-base` |
| 训练方式 | 全参数微调 |
| 数据 | final train 3,042 / final dev 660 |
| Epochs | 3 |
| Batch size | 64 |
| Max length | 64 |
| Learning rate | `5e-5` |
| Dropout | 0.1 |
| Warmup | 1% |
| Weight decay | `1e-5` |
| Seed | 42 |
| 本地 GPU | RTX 5060 Laptop 8 GB |
| Test | 未运行 |

### 7.2 每轮结果

| Epoch | Train loss | Weighted F1 | SoftCons | HardCons |
|---:|---:|---:|---:|---:|
| 1 | 1.3005 | 0.3784 | 0.2744 | 0.1913 |
| 2 | 1.0400 | 0.5986 | 0.8123 | 0.6426 |
| 3 | 0.6947 | **0.6925** | **0.7762** | **0.6606** |

最终 dev：

```text
Accuracy       0.6924
Macro F1       0.6757
Weighted F1    0.6925
SoftCons       0.7762
HardCons       0.6606
```

### 7.3 Epoch 3 标签指标

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| BACKWARD_ENTAILMENT | 0.7296 | 0.7526 | 0.7409 |
| EQUIVALENCE | 0.6813 | 0.6264 | 0.6527 |
| FORWARD_ENTAILMENT | 0.7592 | 0.7632 | 0.7612 |
| NEGATIVE_OTHER | 0.5310 | 0.5660 | 0.5479 |

### 7.4 结论

- 三个 epoch 内 dev F1 持续上升；
- epoch 2 到 epoch 3 的 F1/HardCons 上升，但 SoftCons 下降；
- 该模型是当前综合 F1/SoftCons 最好的系统；
- `NEGATIVE_OTHER` 和 `EQUIVALENCE` 仍是最弱类别。

## 8. E3：DeBERTa-v2-xxlarge 早期失败实验

### 8.1 基座与训练方式

- 模型：`microsoft/deberta-v2-xxlarge`；
- 参数量：约 1.574B；
- 框架：Transformers + PEFT LoRA；
- Unsloth：未使用，因为其主接口面向受支持的 decoder LLM，不能直接加载该 encoder 分类模型；
- QLoRA：在 xxlarge 最终有效实验中未使用。

### 8.2 LoRA 配置

```text
r = 16
alpha = 32
dropout = 0.05
target_modules = query_proj,value_proj
task_type = SEQ_CLS
```

标准 LoRA 实验还训练：

```text
classifier
pooler
```

### 8.3 环境问题

#### 问题 A：Kaggle torchao 版本冲突

现象：

```text
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
```

发生位置：

```text
get_peft_model()
```

原因：

- Kaggle 预装旧版可选包 `torchao 0.10.0`；
- 较新的 PEFT 在挂 LoRA 时检查 torchao；
- 该实验不需要 torchao。

处理：

```bash
pip uninstall -y torchao
```

结果：

- 模型下载没有浪费，权重继续缓存；
- 卸载后 LoRA 可以正常创建；
- 这是环境依赖问题，不是模型或数据问题。

#### 问题 B：纯 FP16 权重训练不稳定

早期配置：

```text
compute_dtype = float16
lr = 1e-4
```

结果：

| Epoch | Weighted F1 | SoftCons | HardCons |
|---:|---:|---:|---:|
| 1 | 0.2719 | 0.1119 | 0.0686 |
| 2 | 0.1287 | 0.0000 | 0.0000 |
| 3 | 0.1287 | 0.0000 | 0.0000 |

判断：

- epoch 2 开始预测塌缩为单一类别；
- `0.1287` 与“大多数/all-forward 判断”的量级一致；
- SoftCons/HardCons 降为 0，说明没有形成方向兼容预测；
- 不是正常收敛。

处理：

- 主实验改为 FP32 权重；
- 学习率从 `1e-4` 降至 `2e-5`；
- 后续局部代码增加了 FP32 主权重 + autocast + GradScaler 的稳定 AMP 方案；
- 注意：最终 E4 产物实际使用的是纯 FP32 计算，当时 AMP 修改尚未推送到 Kaggle 代码。

#### 问题 C：随机 pooler 被冻结

加载日志显示：

```text
pooler.dense.weight MISSING
pooler.dense.bias   MISSING
classifier.weight   MISSING
classifier.bias     MISSING
```

原因是 `microsoft/deberta-v2-xxlarge` checkpoint 是 masked-LM 架构，没有任务专用 pooler/classifier。

错误状态：

- 只把 `classifier` 加入 `modules_to_save`；
- 随机初始化的 `pooler` 被 PEFT 冻结；
- 分类器长期接在随机 pooled 表示上。

中间诊断实验使用 FP32、`lr=2e-5`，但 pooler 仍冻结：

| Epoch | Weighted F1 | SoftCons | HardCons |
|---:|---:|---:|---:|
| 1 | 0.2907 | 0.3105 | 0.1155 |
| 2 | 0.3316 | 0.3249 | 0.1661 |
| 3 | 0.3402 | 0.2996 | 0.1661 |

处理：

```python
modules_to_save=["classifier", "pooler"]
```

修复后 trainable parameters：

```text
7,085,572 / 1,574,002,184 = 0.4502%
```

参数构成近似为：

```text
query_proj LoRA   ~2.36M
value_proj LoRA   ~2.36M
pooler            ~2.36M
classifier        ~0.006M
```

## 9. E4：有效的 DeBERTa-v2-xxlarge LoRA

### 9.1 实际训练配置

| 项目 | 值 |
|---|---|
| 模型 | `microsoft/deberta-v2-xxlarge` |
| 架构 | DeBERTa-v2 encoder + sequence classification head |
| 训练方式 | PEFT LoRA，非 QLoRA |
| 主权重精度 | FP32 |
| 输入 | `text1 [SEP] text2` |
| Train | 3,042 |
| Dev | 660 |
| Epochs | 3 |
| Micro batch | 1 |
| Gradient accumulation | 16 |
| Effective batch | 16 |
| Max length | 128 |
| Learning rate | `2e-5` |
| Warmup | 3% |
| Weight decay | `1e-5` |
| Dropout | 0.1 |
| Gradient checkpointing | enabled |
| Seed | 42 |
| Deterministic | disabled |
| GPU | Kaggle Tesla T4，单卡执行 |
| LoRA r | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA targets | `query_proj,value_proj` |
| Modules to save | `classifier,pooler` |

### 9.2 每轮结果

| Epoch | Weighted F1 | SoftCons | HardCons |
|---:|---:|---:|---:|
| 1 | 0.5006 | 0.4007 | 0.3466 |
| 2 | 0.6598 | 0.7401 | 0.6606 |
| 3 | **0.6802** | **0.7437** | **0.6751** |

### 9.3 Epoch 3 派生指标

```text
Accuracy       0.6970
Weighted F1    0.6802
SoftCons       0.7437
HardCons       0.6751
```

SoftCons 和 HardCons 的评价单位是 277 个可逆 source pairs。

### 9.4 Epoch 3 标签指标

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| BACKWARD_ENTAILMENT | 0.7130 | 0.8368 | 0.7700 |
| EQUIVALENCE | 0.7202 | 0.6954 | 0.7076 |
| FORWARD_ENTAILMENT | 0.6833 | 0.7947 | 0.7348 |
| NEGATIVE_OTHER | 0.6042 | 0.2736 | 0.3766 |

### 9.5 Confusion matrix

| Gold \ Pred | BACKWARD | EQUIVALENCE | FORWARD | NEGATIVE |
|---|---:|---:|---:|---:|
| BACKWARD | 159 | 15 | 11 | 5 |
| EQUIVALENCE | 17 | 121 | 27 | 9 |
| FORWARD | 16 | 18 | 151 | 5 |
| NEGATIVE | 31 | 14 | 32 | 29 |

`NEGATIVE_OTHER` 的关键问题：

```text
106 个 NEGATIVE_OTHER 中只有 29 个正确
77 个被预测为 BACKWARD/EQUIVALENCE/FORWARD
recall = 27.36%
```

这说明模型学到了方向关系，但将大量相关或不完全相同短语过度解释为可逆 NLI 关系。

## 10. E2 与 E4 对比

| 模型 | Weighted F1 | SoftCons | HardCons |
|---|---:|---:|---:|
| ModernBERT-base full FT | **0.6925** | **0.7762** | 0.6606 |
| DeBERTa-v2-xxlarge LoRA | 0.6802 | 0.7437 | **0.6751** |

差值：

```text
DeBERTa Weighted F1：-0.0123
DeBERTa SoftCons：  -0.0325
DeBERTa HardCons：  +0.0144
```

当前不能声称 DeBERTa 全面优于 ModernBERT。

更准确的论文表述是：

- DeBERTa LoRA 取得更高 HardCons；
- ModernBERT 全参数微调取得更高 Weighted F1 和 SoftCons；
- 两个系统体现了不同的准确性/双向联合正确性权衡。

## 10.1 E5：NEGATIVE_OTHER 类别加权失败实验

### 实验目的

E4 的 `NEGATIVE_OTHER` 只有：

```text
precision = 0.6042
recall    = 0.2736
F1        = 0.3766
```

因此保持其他条件不变，只提高 `NEGATIVE_OTHER` 的交叉熵权重：

```text
BACKWARD         1.0
EQUIVALENCE      1.0
FORWARD          1.0
NEGATIVE_OTHER   1.5
```

### 配置

| 项目 | 值 |
|---|---|
| 代码 | commit `d13c1d8` |
| 模型 | `microsoft/deberta-v2-xxlarge` |
| 训练方式 | 标准 LoRA，FP32 |
| Micro batch | 1 |
| Gradient accumulation | 16 |
| Effective batch | 16 |
| Max length | 128 |
| Learning rate | `2e-5` |
| Epochs | 3 |
| LoRA r / alpha | 16 / 32 |
| LoRA targets | `query_proj,value_proj` |
| Modules to save | `classifier,pooler` |
| NEGATIVE loss weight | 1.5 |
| 运行方式 | Kaggle Save & Run All，1 小时 45 分 |

### 每轮结果

| Epoch | Weighted F1 | SoftCons | HardCons |
|---:|---:|---:|---:|
| 1 | 0.4001 | 0.4116 | 0.2924 |
| 2 | 0.4289 | 0.4260 | 0.3321 |
| 3 | **0.4395** | **0.4404** | **0.3502** |

### Epoch 3 标签指标

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| BACKWARD_ENTAILMENT | 0.4579 | 0.7158 | 0.5585 |
| EQUIVALENCE | 0.5464 | 0.5747 | 0.5602 |
| FORWARD_ENTAILMENT | 0.4693 | 0.4421 | 0.4553 |
| NEGATIVE_OTHER | 0.0000 | 0.0000 | 0.0000 |

106 个 `NEGATIVE_OTHER` 样本的预测为：

```text
NEGATIVE -> BACKWARD     38
NEGATIVE -> EQUIVALENCE  27
NEGATIVE -> FORWARD      41
NEGATIVE -> NEGATIVE      0
```

整个 dev 只有 1 条样本被预测为 `NEGATIVE_OTHER`。

### 结论

- 提高 NEGATIVE 权重没有让模型更多预测 NEGATIVE；
- 训练反而明显失稳，三个主指标均大幅下降；
- NEGATIVE F1 变为 0；
- 该结果不能用于论文主张，也不能替换 E4；
- 在缺少完整 loss 日志时，不能把原因简单归为“类别权重过大”；
- 当前应停止继续增大类别权重。

### 产物

```text
D:\下载\dico_nli_deberta_xxlarge_lora_nweight15_fp32.zip
```

大小：

```text
112,028,610 bytes
```

SHA256：

```text
9d084b367069b5ebff8ba65d1d334a845be9b060f55107a2b33087faf0807884
```

Adapter 配置与原模型一致，`classifier` 和 `pooler` 仍正确保存在 `modules_to_save` 中。

## 10.2 E6：E4 dev logit 校准

### 实验目的

不重新训练 E4，只加载其 epoch 3 adapter，对 dev logits 的
`NEGATIVE_OTHER` 分量增加偏置，研究是否能改善 N 类别，同时观察三项指标的权衡。

### 方法

- 基座：`microsoft/deberta-v2-xxlarge`
- adapter：E4 `checkpoints/epoch_3`
- 推理精度：FP16
- 输入：`text1 [SEP] text2`
- 最大长度：128
- 扫描偏置：`-2.0` 至 `4.0`
- 不使用 dev 标签训练模型，只用 dev 选择后处理偏置

校准流程保存了：

```text
dev_logits.npy
每种偏置的 prediction CSV
bias_results.csv
bias_results.json
```

### 关键候选

| NEGATIVE bias | Weighted F1 | SoftCons | HardCons | N recall | N F1 | 解释 |
|---:|---:|---:|---:|---:|---:|---|
| -2.00 | 0.6194 | **0.7581** | **0.6895** | 0.0283 | 0.0545 | 偏向方向关系 |
| -1.00 | 0.6478 | 0.7581 | 0.6895 | 0.1132 | 0.1983 | 方向一致性较好 |
| -0.50 | 0.6647 | 0.7509 | 0.6823 | 0.1887 | 0.2941 | 小幅偏向方向关系 |
| 0.00 | 0.6825 | 0.7437 | 0.6751 | 0.2830 | 0.3871 | 最均衡 |
| +0.25 | 0.6893 | 0.7292 | 0.6679 | 0.3396 | 0.4211 | 轻微偏重 F1 |
| +0.75 | **0.6953** | 0.6823 | 0.6245 | 0.4717 | 0.4673 | F1 最大 |
| +1.00 | 0.6935 | 0.6426 | 0.5884 | 0.5472 | 0.4677 | 继续偏重 N |
| +2.00 | 0.6293 | 0.4765 | 0.4440 | 0.7453 | 0.4147 | 过度预测 N |

### 结论

- 不存在同时提高三项指标的偏置；
- Weighted F1 的高点位于 `+0.75`；
- HardCons/SoftCons 的高点位于负偏置；
- 无校准 `bias=0` 仍是三项最均衡的运行；
- `+0.25` 适合作为 balanced/F1-oriented secondary run；
- `+0.75` 适合作为 maximum-F1 secondary run；
- `-1.0` 适合作为 consistency-oriented secondary run；
- `-1.0` 在所有三项上支配 `-2.0`，因此不推荐 `-2.0`；
- 不能把自动按 F1 选择的 `+0.75` 描述为总体最佳系统。

FP16 推理得到的 baseline 为：

```text
Weighted F1 = 0.6825
SoftCons    = 0.7437
HardCons    = 0.6751
```

与 E4 原始 FP32 结果 `0.6802 / 0.7437 / 0.6751` 基本一致，F1 的少量差异来自推理精度不同。

### 产物

```text
D:\下载\dico_nli_e4_bias_calibration.zip
```

大小：

```text
54,347 bytes
```

SHA256：

```text
fa364c9695c6a6add430751679c88c70249858f60b25b081e293be1606391214
```

## 11. 产物

### 11.1 当前最佳 DeBERTa adapter

Kaggle 输出目录：

```text
/kaggle/working/dico_nli_runs/deberta_v2_xxlarge_lora_track1_fp32_pooler
```

最佳 checkpoint：

```text
checkpoints/epoch_3
```

下载后的完整压缩包：

```text
D:\下载\dico_nli_deberta_xxlarge_lora_fp32_pooler.zip
```

文件大小：

```text
112,058,156 bytes
```

SHA256：

```text
17fade2c8c8d30b8abaa2a0cbd029902ba53b6211f5ec183b14d3d8a4380467f
```

已验证 adapter 配置：

```json
{
  "base_model_name_or_path": "microsoft/deberta-v2-xxlarge",
  "task_type": "SEQ_CLS",
  "r": 16,
  "lora_alpha": 32,
  "target_modules": ["query_proj", "value_proj"],
  "modules_to_save": ["classifier", "pooler"]
}
```

压缩包不包含基座模型，只包含 adapter、tokenizer、三个 epoch 的 checkpoint、dev 预测和评分。

### 11.2 ModernBERT 输出

```text
starter_kit/results/baseline_modernbert_track1
```

## 12. 代码修改记录

### 2026-10-27

- 将双句 tokenizer 输入改为 `text1 [SEP] text2`；
- 修复 Windows 下标签 `int32` 与交叉熵 `int64` 不兼容的问题；
- 使用 `LongTensor` 标签。

### 2026-10-28

- 支持任意 batch size 1–64；
- 加入 gradient accumulation；
- 加入 PEFT LoRA；
- 加入可选 4-bit QLoRA；
- 支持分类头/adapter 保存和重新加载；
- 关闭 4-bit 模式下的确定性算法约束；
- 每个 epoch 保存 LoRA checkpoint；
- 清理 English-only 仓库内容。

### 2026-10-31 至 2026-11-01

- 修复 Kaggle torchao 冲突；
- 将 DeBERTa LoRA 改为更稳定的 FP32 主权重；
- 为后续代码加入 FP16/BF16 autocast 与 GradScaler；
- 将 `pooler` 加入可训练模块；
- 当前最佳 adapter 来自 pooler 修复后的 E4。

### Git 状态

重要提交：

```text
f81fe16  English-only project snapshot
9632652  Kaggle LoRA support
3b73d60  torchao compatibility note
5823589  DeBERTa LoRA stability/Amp code
9d5eb75  make DeBERTa pooler trainable
```

注意：

- 最后两个提交在撰写本文档时尚未成功推送到 GitHub；
- 当时 GitHub HTTPS 连接被反复重置；
- 最终 Kaggle 运行使用远程代码加 Notebook 内运行时补丁；
- 复现 E4 时必须同时记录运行时 patch，不能只看远程 commit。

## 13. 已知缺口

目前仍未保存或尚待确认：

- DeBERTa 最终运行的最佳 epoch 训练日志；
- 每个 epoch 的 train loss；
- 峰值 CUDA 显存；
- 完整 Kaggle 环境包版本；
- official test prediction；
- 多随机种子实验；
- official test prediction；
- N 类别的结构性改进方法。

## 14. 下一组优先实验

### E5：NEGATIVE_OTHER 类别加权

状态：已运行，结果失败，已拒绝。

保持 E4 其他条件不变，只改变类别权重：

```text
BACKWARD         1.0
EQUIVALENCE      1.0
FORWARD          1.0
NEGATIVE_OTHER   1.5
```

实现：

```text
--negative_class_weight 1.5
```

训练损失改为带类别权重的交叉熵；其他类别权重保持 1.0。

计划输出目录：

```text
/kaggle/working/dico_nli_runs/deberta_v2_xxlarge_lora_track1_nweight15
```

实际结果：见 10.1。不要再重复 `NEGATIVE_OTHER=1.5` 的该配置。

评价标准：

- NEGATIVE F1 是否从 0.3766 明显提升；
- Weighted F1 是否超过 E4 的 0.6802；
- SoftCons、HardCons 是否明显退化；
- 是否达到或超过 ModernBERT 的 Weighted F1 0.6925。

### 后续候选

- 将 `bias=0` 作为主系统；
- 将 `bias=+0.25` 作为平衡变体，将 `bias=+0.75` 作为 F1 最大变体；
- 将 `bias=-1.0` 作为 consistency-oriented variant；
- 保留 E6 校准产物作为误差分析材料；
- 探索单独的 reversible/negative 二阶段判定器；
- 探索 pair-level consistency 约束；
- 增加 LoRA target modules；
- 使用更多随机种子；
- 保存完整训练日志和显存统计。

## 15. 论文写作注意事项

可以写：

- 使用 `text1 [SEP] text2` 单序列编码；
- 比较 encoder 全参数微调与 encoder LoRA；
- 报告三项指标而非只报告 F1；
- 讨论 HardCons 与 F1/SoftCons 的权衡；
- 讨论 `NEGATIVE_OTHER` 是异质兜底类别，容易被相关短语误导。

不能写：

- 不能写最终 DeBERTa 实验使用了 Unsloth；
- 不能写 E4 使用了 QLoRA；
- 不能把 trial 0.8372 当作正式 dev 成绩；
- 不能只因为 Weighted F1 较高就声称方向一致性更好；
- 不能声称 DeBERTa LoRA 已全面超过 ModernBERT；
- 不能把缺失的 pooler/classifier 初始化为模型加载错误。
- 不能把 E5 的 NEGATIVE F1=0 归因于类别权重不足；
- 不能继续使用 NEGATIVE weight 1.5 的 E5 adapter。
- 不能把 E6 的 `bias=+0.75` 描述为三项指标总体最优；
- 不能把 dev 上选择的偏置描述为在隐藏 test 上已验证有效。

