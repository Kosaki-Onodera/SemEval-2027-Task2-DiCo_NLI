# SemEval 2027 Task 2 英文赛道任务说明

> 文档范围：English track（Track 1）
>
> 整理日期：2026-10-28
> 依据：仓库根目录 `README.md`、`final_data/README.md`、`trial_data/README.md`、`evaluation_functions/` 和 starter kit 中的正式实现。

## 1. 一句话理解任务

给定一个**有顺序的英文短语对** `(text1, text2)`，判断二者之间属于以下哪一种关系：

- `EQUIVALENCE`：双向等价；
- `FORWARD_ENTAILMENT`：`text1` 蕴含 `text2`；
- `BACKWARD_ENTAILMENT`：`text2` 蕴含 `text1`；
- `NEGATIVE_OTHER`：不属于上述三种可逆方向关系。

DiCo-NLI 不只是普通的四分类 NLI。它还会把同一个底层短语对交换顺序，检查模型在 `(text1, text2)` 和 `(text2, text1)` 两个方向上的预测是否符合确定性的标签反转关系。

因此，一个系统需要同时做到两件事：

1. 对单个有序实例给出正确标签；
2. 对同一短语对的两个方向给出彼此方向兼容的标签。

## 2. 这个任务评估什么

### 2.1 “有方向”的含义

标签描述的是 `text1` 和 `text2` 在当前顺序下的关系，而不是两个短语之间一个无方向的相似度。

例如：

| text1 | text2 | 含义 | 标签 |
|---|---|---|---|
| `A black dog` | `A dog` | 黑狗是狗，因此 `text1` 蕴含 `text2` | `FORWARD_ENTAILMENT` |
| `A dog` | `A black dog` | 狗不一定是黑狗，但原关系反向成立 | `BACKWARD_ENTAILMENT` |

同一个底层短语对交换顺序后，前向和后向标签必须互换。

### 2.2 为什么强调方向一致性

普通 NLI 通常只评价单个有序样本。一个模型可能在 `(A, B)` 上预测前向蕴含，却在 `(B, A)` 上仍然预测前向蕴含。

这种错误有两种层面的含义：

- 从单样本分类角度看，它可能在某一边碰巧正确；
- 从完整短语对角度看，它没有形成前后一致的方向关系。

DiCo-NLI 将这两件事拆开评价：

- 分类是否正确；
- 两个方向的预测是否自洽；
- 两个方向是否同时正确。

### 2.3 与 Reversal Curse 的区别

本任务受到 Reversal Curse 的启发，但**不是**对语言模型参数化事实记忆的直接测试。

Reversal Curse 的典型问题是：模型见过 `A is B`，是否就能回答 `B is A`。

DiCo-NLI 在每次判断时都会同时提供两个短语，测试的是模型对有序语义关系的判断，以及交换顺序后是否保持一致。任务说明明确强调两者不能混为一谈。

## 3. 四个标签的精确定义

### 3.1 标签总表

| 标签 | 当前方向下的含义 | 交换顺序后的标签 | 是否参与一致性指标 |
|---|---|---|---|
| `EQUIVALENCE` | `text1` 和 `text2` 双向等价 | `EQUIVALENCE` | 是 |
| `FORWARD_ENTAILMENT` | `text1` 蕴含 `text2` | `BACKWARD_ENTAILMENT` | 是 |
| `BACKWARD_ENTAILMENT` | `text2` 蕴含 `text1` | `FORWARD_ENTAILMENT` | 是 |
| `NEGATIVE_OTHER` | 不属于上述三种可逆关系 | 无确定的统一反向标签 | 否 |

注意标签名称的方向：

- **FORWARD** 表示信息沿 `text1 -> text2` 方向成立；
- **BACKWARD** 表示信息沿 `text2 -> text1` 方向成立；
- 不能只根据短语的长短判断方向。

### 3.2 `EQUIVALENCE`

两个短语在当前语境下可以互相推出，或者表达相同的核心含义。

交换 `text1` 和 `text2` 后，标签仍然是 `EQUIVALENCE`。

### 3.3 `FORWARD_ENTAILMENT`

如果接受 `text1` 表达的含义，就能推出 `text2`；反过来不一定成立。

交换顺序后，它必须变成 `BACKWARD_ENTAILMENT`。

### 3.4 `BACKWARD_ENTAILMENT`

如果接受 `text2` 表达的含义，就能推出 `text1`；反过来不一定成立。

交换顺序后，它必须变成 `FORWARD_ENTAILMENT`。

### 3.5 `NEGATIVE_OTHER`

这是一个**广义兜底标签**，不能简单等同于“矛盾”。

它覆盖 PhrasIS 原始标注中不属于可逆正关系集合的项目，可能包括：

- 相反或冲突关系；
- 相关但不等价；
- 只有一般相似性的关系；
- 无明确蕴含关系的短语；
- 其他无法稳定映射到三种可逆标签的情况。

`NEGATIVE_OTHER` 有两层重要含义：

1. 它只在四分类 Weighted F1 中计分；
2. 它不参与 SoftCons 和 HardCons，因为官方没有为它定义一个统一的反向标签。

因此，遇到 `NEGATIVE_OTHER` 时，不应自行假定“交换顺序后还是 `NEGATIVE_OTHER`”。

## 4. 数据的底层单位：source phrase pair

### 4.1 什么是 source pair

数据来自 PhrasIS 的短语级 NLI 标注，再从中抽取可逆部分。一个 source phrase pair 可以理解为底层的一对短语，例如：

```text
A black dog
A dog
```

对于可逆关系，同一个 source pair 会派生出两个有序实例：

```text
(A black dog, A dog)      -> FORWARD_ENTAILMENT
(A dog, A black dog)      -> BACKWARD_ENTAILMENT
```

这两行共享同一个 `pair_id`，并通过 `reverse_pair_id` 互相指向。

### 4.2 不应该把 `pair_id` 当成单个标签

`pair_id` 表示底层短语对所属的组，不是分类标签。一个 `pair_id` 下的两个方向通常具有不同标签。

正确理解是：

- `instance_id` 对应一个有序预测实例；
- `pair_id` 把反向变体组织在一起；
- `label` 描述这个 `instance_id` 当前方向上的关系。

### 4.3 是否是完整句子

不是。数据包含名词短语、动词片段、介词短语、数量表达、图像描述片段和新闻标题片段。

例如：

```text
Two brown dogs
Two dogs
```

```text
in front of a pool
at a pool
```

```text
at least 20 injured
at least 20 people injured
```

这些片段有时单独看不完整，但它们是 PhrasIS/iSTS 对齐框架中的短语语义单元。任务中的“compositional”主要指这种短语级语义组合，不是任意模板层面的组合泛化测试。

## 5. English track 的公开数据

### 5.1 当前仓库范围

官方任务设计了四个赛道：

1. English；
2. Spanish；
3. Basque；
4. Mixed multilingual。

但当前工作区只保留了 English track。`final_data/README.md` 和 `trial_data/README.md` 都明确说明西班牙语、巴斯克语和混合语言文件已因“只使用英语赛道”而从本地工作区移除。

因此，当前项目实际应聚焦 Track 1。

### 5.2 数据规模

| Split | 实例数 | 可逆 source pairs | 可逆实例数 | `NEGATIVE_OTHER` |
|---|---:|---:|---:|---:|
| Train | 3,042 | 1,282 | 2,564 | 478 |
| Dev | 660 | 277 | 554 | 106 |

这里的区别很重要：

- Weighted F1 在 3,042 或 660 个**有序实例**上计算；
- SoftCons 和 HardCons 在 1,282 或 277 个**可逆 source pairs**上计算。

### 5.3 标签分布

| Split | `BACKWARD_ENTAILMENT` | `EQUIVALENCE` | `FORWARD_ENTAILMENT` | `NEGATIVE_OTHER` | 合计 |
|---|---:|---:|---:|---:|---:|
| Train | 881 | 802 | 881 | 478 | 3,042 |
| Dev | 190 | 174 | 190 | 106 | 660 |

前向和后向实例数量相同，是因为可逆 source pair 会被翻转成两个方向。`EQUIVALENCE` 在每个 source pair 的两个方向上都是 `EQUIVALENCE`，所以其 source pair 数约等于实例数的一半。

### 5.4 Trial data

英语 trial 数据共 797 个有序实例，来自 461 个 source pairs：

| 类型 | Source pairs 或实例数 |
|---|---:|
| `BACKWARD_ENTAILMENT` source pairs | 125 |
| `EQUIVALENCE` source pairs | 86 |
| `FORWARD_ENTAILMENT` source pairs | 125 |
| `NEGATIVE_OTHER` 单方向实例 | 125 |
| 合计有序实例 | 797 |

Trial 数据用于跑通训练和提交格式，不应被当作正式 dev 集选择模型。

### 5.5 数据切分原则

官方数据按底层 source phrase pair 分组切分。因此：

- 原顺序和反顺序不会跨 train/dev/test；
- 派生变体不会泄漏到另一个 split；
- 不应再按单行随机切分后把一半方向放进训练、另一半放进验证。

如果为了本地实验而重新抽样，至少要按 `pair_id` 分组划分。

## 6. 文件结构与字段

### 6.1 训练和开发文件

当前 English track 位于：

```text
final_data/
├── train/
│   ├── dico_nli_train_track1_participant_labeled.csv
│   ├── dico_nli_train_track1_reference.csv
│   └── dico_nli_train_track1_submission_template.csv
└── dev/
    ├── dico_nli_dev_track1_participant_labeled.csv
    ├── dico_nli_dev_track1_reference.csv
    └── dico_nli_dev_track1_submission_template.csv
```

### 6.2 `participant_labeled.csv`

列包括：

```text
instance_id,pair_id,text1_lang,text2_lang,text1,text2,label
```

用途：

- 读取输入文本；
- 使用公开标签进行训练；
- 在 dev 上计算本地分类结果。

`text1_lang` 和 `text2_lang` 在 English track 中均为 `en`。

### 6.3 `reference.csv`

列包括：

```text
instance_id,pair_id,text1_lang,text2_lang,text1,text2,reverse_pair_id,label
```

与 participant 文件的关键差别是多了：

```text
reverse_pair_id
```

它用于告诉官方 scorer 哪两个有序实例组成一个可逆 source pair。

不要把 test 阶段的参与者输入预设为一定包含 `reverse_pair_id`。它主要存在于公开 reference/gold 文件中。

### 6.4 `submission_template.csv`

提交文件必须恰好包含：

```csv
instance_id,label
```

要求：

- 每个评价实例恰好一行；
- `instance_id` 必须与输入完全一致；
- 不得重复；
- 不得缺少；
- 不得加入输入中不存在的 ID；
- `label` 必须是四个官方标签之一；
- 不能为空；
- 不要额外增加概率、解释或第三列。

## 7. 模型实际接收和输出的内容

### 7.1 官方任务层面

官方要求的输入是一个有序二元组：

```text
(text1, text2)
```

顺序具有语义。交换两个文本会改变 `FORWARD_ENTAILMENT` 和 `BACKWARD_ENTAILMENT`。

官方没有强制要求某个具体模型必须怎样编码。常见做法包括：

- 使用 encoder 的双句输入；
- 手动拼接成一个字符串；
- 使用提示词构造 decoder 输入；
- 使用双塔或交互式模型。

### 7.2 当前 starter kit 的编码方式

当前 starter kit 已将输入改成单段拼接：

```text
text1 <SEP> text2
```

其中 `<SEP>` 优先使用 tokenizer 自带的 `sep_token`，例如 DeBERTa 的 `[SEP]`。

这是一种实现选择，不是比赛的新增输入格式。它的效果是：

- 模型仍能看到两个短语及其顺序；
- tokenizer 使用单文本流程；
- `token_type_ids` 通常全为 0；
- 模型不能依赖双句 segment embedding 区分两个片段，只能从文本内容、分隔符和位置中学习方向。

### 7.3 分类输出

每个有序实例输出一个标签，不做“无关系二分类”和“方向二分类”的隐式替换。

最终文件仍然需要合并成四分类标签：

```text
EQUIVALENCE
FORWARD_ENTAILMENT
BACKWARD_ENTAILMENT
NEGATIVE_OTHER
```

## 8. 官方评价指标

English track 的官方结果是一个三项组合：

```text
(Weighted F1, SoftCons, HardCons)
```

没有把三项压缩成一个唯一的总分。

### 8.1 Weighted F1

Weighted F1 看单实例四分类质量。

对每个标签 `l`：

```text
Precision_l = TP_l / (TP_l + FP_l)
Recall_l    = TP_l / (TP_l + FN_l)
F1_l        = 2 * Precision_l * Recall_l / (Precision_l + Recall_l)
```

最终：

```text
Weighted F1 = sum(support_l / N * F1_l)
```

特点：

- 包含全部四个标签；
- 包含 `NEGATIVE_OTHER`；
- 按每个标签的 gold support 加权；
- 评价单位是**单个有序实例**。

### 8.2 反向操作符 Rev

定义：

```text
Rev(EQUIVALENCE)          = EQUIVALENCE
Rev(FORWARD_ENTAILMENT)   = BACKWARD_ENTAILMENT
Rev(BACKWARD_ENTAILMENT)  = FORWARD_ENTAILMENT
```

`Rev(NEGATIVE_OTHER)` 没有官方定义，因此这种标签不进入方向一致性指标。

### 8.3 SoftCons

SoftCons 衡量同一模型对同一可逆 source pair 的两个预测是否方向兼容。

设：

```text
ŷ = 对 (A, B) 的预测
ŷr = 对 (B, A) 的预测
```

则：

```text
SoftCons(source pair) =
  1, 如果 ŷ 和 ŷr 都是可逆标签，且 ŷr = Rev(ŷ)
  0, 否则
```

最终对全部可逆 source pairs 取平均。

SoftCons 的特点是：

- **不检查答案是否正确**；
- 只检查两个预测之间是否方向兼容；
- 一个系统可能非常自洽，但完全答错；
- 任一方向预测为 `NEGATIVE_OTHER` 时，该 pair 的 SoftCons 为 0。

### 8.4 HardCons

HardCons 衡量两个方向是否都答对。

设当前方向 gold label 为：

```text
y
```

反方向 gold label 必为：

```text
Rev(y)
```

则：

```text
HardCons(source pair) =
  1, 如果 ŷ = y 且 ŷr = Rev(y)
  0, 否则
```

最终对全部可逆 source pairs 取平均。

特点：

- 评价单位是完整可逆短语对；
- 两个方向必须同时正确；
- 只答对一个方向得 0；
- HardCons 永远不可能高于 SoftCons。

## 9. SoftCons 与 HardCons 的直观例子

假设 gold 关系为：

```text
(A, B) = FORWARD_ENTAILMENT
(B, A) = BACKWARD_ENTAILMENT
```

不同预测结果如下：

| `(A, B)` 预测 | `(B, A)` 预测 | SoftCons | HardCons | 解释 |
|---|---|---:|---:|---|
| `FORWARD_ENTAILMENT` | `BACKWARD_ENTAILMENT` | 1 | 1 | 两方向都正确 |
| `EQUIVALENCE` | `EQUIVALENCE` | 1 | 0 | 自洽，但两个方向都错 |
| `FORWARD_ENTAILMENT` | `FORWARD_ENTAILMENT` | 0 | 0 | 第一边正确，但方向不兼容 |
| `BACKWARD_ENTAILMENT` | `FORWARD_ENTAILMENT` | 1 | 0 | 两方向都反了，但彼此自洽 |
| `FORWARD_ENTAILMENT` | `NEGATIVE_OTHER` | 0 | 0 | 有一个方向离开可逆标签空间 |

这组例子说明为什么不能只看 Weighted F1：

- 高 F1 可能伴随方向不稳定；
- 高 SoftCons 可能只是稳定地犯同一种系统性错误；
- HardCons 才要求完整短语对的两个方向同时正确。

## 10. 指标之间应该怎样联合解读

### 10.1 高 F1、低 SoftCons

说明单样本分类可能不错，但模型对交换顺序敏感。常见原因：

- 把标签方向学反；
- 依赖 `text1` 或 `text2` 的位置偏置；
- 对“更具体 / 更一般”的方向判断不稳定；
- 训练时没有充分覆盖反向模式。

### 10.2 高 SoftCons、低 HardCons

说明模型在反向上稳定，但整体关系判断存在系统偏差。

例如模型经常把前向蕴含和反向蕴含都预测成 `EQUIVALENCE`，会得到很高的 SoftCons，但 HardCons 仍然很低。

### 10.3 Weighted F1 高、HardCons 相对低

说明很多单实例答对了，但关键可逆 source pair 没有两个方向同时答对。HardCons 对短语对质量的惩罚更严格。

### 10.4 三项指标的用途

- Weighted F1：整体四分类正确性；
- SoftCons：方向稳定性；
- HardCons：双向联合正确性。

模型选择时应同时观察三项，而不是只按 Weighted F1 挑 checkpoint。

## 11. 简单基线透露出的信息

在英语 trial 数据上：

| 系统 | Weighted F1 | SoftCons | HardCons |
|---|---:|---:|---:|
| Random | 0.2696 | 0.2083 | 0.0774 |
| 全部预测多数类 | 0.1498 | 0.0000 | 0.0000 |

多数类策略的 SoftCons 和 HardCons 都是 0，因为它无法形成可逆的方向关系。

Random 的 SoftCons 约为 0.208，HardCons 约为 0.077，说明随机预测偶尔也能碰巧形成兼容或双方向正确的结果。

Trial 数据只用于格式和流程验证，不能把这些数值当成正式 dev 排名。

## 12. 常见理解陷阱

### 12.1 把它当成 STS 或相似度分类

两个短语“相似”不足以确定四分类。

例如，“更具体”和“更一般”的两个短语可能高度相似，但仍具有明确的前向或后向蕴含关系。

### 12.2 把 `NEGATIVE_OTHER` 只理解成“相反”

它覆盖多种关系，是一个大兜底类别。模型不应把所有低相似度样本机械地当作 contradiction。

### 12.3 忽略 `text1` 和 `text2` 的顺序

同一对文本交换顺序后，前向和后向标签必须互换。

### 12.4 把 `pair_id` 直接当作标签

`pair_id` 是分组 ID。真正的标签在每一行的 `label` 中。

### 12.5 假设每个 `pair_id` 都有两行

可逆正标签通常有 reciprocal pair；`NEGATIVE_OTHER` 则可能只有一个方向，并且 `reverse_pair_id` 为空。

### 12.6 随机按行划分 train/dev

这会泄漏同一 source pair 的反向实例，使 dev 分数偏高。官方 split 按 `pair_id` 分组，本地重新切分也应遵守这一点。

### 12.7 用 test ID 或 `reverse_pair_id` 做训练特征

`reverse_pair_id` 是 gold/reference 结构信息，不应假定在隐藏测试输入中可用。模型训练应主要依赖 `text1` 和 `text2` 的语义。

### 12.8 只优化 Weighted F1

官方不是单一 F1 排行榜。只追求四分类 F1 可能产生大量方向不一致的预测。

### 12.9 把 trial 当正式 dev

Trial 数据用来验证代码和提交格式。正式模型选择应使用 `final_data/dev`。

## 13. 面向 English track 的建模建议

### 13.1 一个合理的起点

对英语任务，encoder-only 模型通常是直接且强的基线，例如：

- DeBERTa-v3；
- ModernBERT；
- RoBERTa/ELECTRA 等中大型英文 encoder。

目标不是必须使用这些模型，而是优先建立稳定的四分类基线。

### 13.2 输入顺序一定要保留

无论使用双句 tokenizer 还是 `text1 [SEP] text2`：

- 不要只按字母或长度排序文本；
- 不要先构造无序特征再分类；
- 交换文本后应重新预测，不能复用原方向的标签。

### 13.3 数据本身已经包含反向实例

公开训练文件中，可逆 source pair 的两个方向通常都已出现。

训练时应保持两侧成对：

- 不要打乱后错误地把同一 pair 的分组关系丢掉；
- 如果做采样或小样本实验，尽量成对采样；
- 评估时报告 F1、SoftCons、HardCons 三项。

### 13.4 训练和验证的目标

建议至少记录：

- dev Weighted F1；
- dev SoftCons；
- dev HardCons；
- per-label precision/recall/F1；
- 混淆矩阵；
- 方向错误实例。

如果两个 checkpoint 的 Weighted F1 接近，优先选择方向一致性更好、且没有严重 `NEGATIVE_OTHER` 召回问题的模型。

### 13.5 小样本或低显存设置

可以使用 LoRA/QLoRA，但这不改变任务评价方式。

轻量实验时应先确认：

- 分类头输出维度为 4；
- tokenizer 的 SEP 行为正确；
- `text1` 和 `text2` 顺序没有被互换；
- dev 提交文件的 ID 与 template 完全一致。

## 14. 推荐的实际实验流程

### 第一步：读取并检查数据

检查：

- 行数；
- 四类标签计数；
- `pair_id` 是否成组；
- `text1`、`text2` 是否为空；
- train 和 dev 的 `pair_id` 是否重叠。

### 第二步：建立可复现的训练/验证切分

直接使用官方 train/dev 文件。

如果需要额外本地验证集，按 `pair_id` 分组切分，而不是按行切分。

### 第三步：训练四分类模型

输入：

```text
text1, text2
```

输出：

```text
EQUIVALENCE
FORWARD_ENTAILMENT
BACKWARD_ENTAILMENT
NEGATIVE_OTHER
```

### 第四步：生成 dev 预测

从 dev 文件读取每个 `instance_id`，为每条记录输出一个标签。

结果文件只保留：

```csv
instance_id,label
```

### 第五步：用官方 scorer 计算三项指标

在仓库根目录运行：

```bash
python -m evaluation_functions \
  --gold final_data/dev/dico_nli_dev_track1_reference.csv \
  --predictions results/my_dev_track1_predictions.csv \
  --output-dir results/dev_track1_scores
```

输出包括：

- `scores.json`：完整结构化报告；
- `scores.txt`：三项主指标的紧凑结果。

同时检查 JSON 中的：

- per-label 指标；
- confusion matrix；
- `pair_errors`；
- reversible pair 数量。

### 第六步：分析方向错误

重点查看 `pair_errors`：

- 两个方向是否都反向；
- 是否经常在两个方向都预测 `FORWARD_ENTAILMENT`；
- 是否经常使用 `EQUIVALENCE` 作为避风港；
- `NEGATIVE_OTHER` 是否挤占了本应可逆的 pair。

### 第七步：只在确认 dev 流程后运行 test

隐藏 test 阶段：

- 不提供 gold label；
- 不提供可用于训练的 `reverse_pair_id`；
- 只提交 `instance_id,label`；
- 不得使用隐藏测试 gold 信息进行开发。

## 15. 一个完整的 English 例子

假设输入两行：

```text
instance_id=A, text1="A black dog", text2="A dog"
instance_id=B, text1="A dog", text2="A black dog"
```

正确金标：

```text
A -> FORWARD_ENTAILMENT
B -> BACKWARD_ENTAILMENT
```

如果模型预测：

```text
A -> EQUIVALENCE
B -> EQUIVALENCE
```

则：

- 两个单实例都分类错误；
- 但两个预测方向兼容；
- SoftCons 为 1；
- HardCons 为 0。

如果模型预测：

```text
A -> FORWARD_ENTAILMENT
B -> BACKWARD_ENTAILMENT
```

则：

- 两个单实例都正确；
- SoftCons 为 1；
- HardCons 为 1。

如果模型预测：

```text
A -> FORWARD_ENTAILMENT
B -> FORWARD_ENTAILMENT
```

则：

- A 单实例正确；
- B 单实例错误；
- 两个方向不兼容；
- SoftCons 为 0；
- HardCons 为 0。

## 16. 提交前自查清单

- [ ] 我只使用 English track 的 train/dev 数据进行正式实验。
- [ ] 我没有按单行随机划分 train/dev 而破坏 source-pair 分组。
- [ ] 我知道 `FORWARD_ENTAILMENT` 表示 `text1` 蕴含 `text2`。
- [ ] 我保留了两个短语的原始顺序。
- [ ] 我的模型输出四个官方标签之一。
- [ ] 我的提交文件恰好只有 `instance_id,label` 两列。
- [ ] 预测文件覆盖全部评价 ID，且没有重复或未知 ID。
- [ ] 我使用官方 scorer 计算了 Weighted F1、SoftCons、HardCons。
- [ ] 我查看了混淆矩阵和 pair-level 错误。
- [ ] 我没有把 SoftCons 高误认为分类正确率高。
- [ ] 我没有把 trial 分数当作正式模型选择依据。
- [ ] 我没有假定 `NEGATIVE_OTHER` 交换顺序后仍有统一反向标签。

## 17. 最需要记住的五句话

1. 这是一个**有顺序**的四分类任务，不是无方向相似度任务。
2. `FORWARD_ENTAILMENT` 和 `BACKWARD_ENTAILMENT` 交换文本顺序后必须互换。
3. `EQUIVALENCE` 自我反转，`NEGATIVE_OTHER` 不参与一致性指标。
4. Weighted F1 看单实例分类，SoftCons 看方向自洽，HardCons 看双向联合正确。
5. 当前仓库只应围绕 English track 开展实验。

## 18. 相关文件索引

- 官方任务总说明：`README.md`
- 当前英语数据说明：`final_data/README.md`
- 英语 trial 说明：`trial_data/README.md`
- 官方评分代码：`evaluation_functions/`
- 标签反转定义：`evaluation_functions/labels.py`
- 指标实现：`evaluation_functions/metrics.py`
- starter kit：`starter_kit/README.md`
- 当前模型输入实现：`starter_kit/src/trainer/helpers/model.py`
