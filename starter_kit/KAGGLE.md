# Kaggle 运行说明

## 输入数据集

数据已上传为 Kaggle Dataset。默认数据根目录为：

```text
/kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data
```

需要的文件：

```text
train/dico_nli_train_track1_participant_labeled.csv
dev/dico_nli_dev_track1_participant_labeled.csv
dev/dico_nli_dev_track1_reference.csv
```

不需要解压或复制数据。

## Notebook 设置

- Accelerator：GPU T4 或 P100；
- Internet：开启，因为需要下载 `microsoft/deberta-v2-xxlarge`；
- 输入：同时添加数据 Dataset 和代码 Dataset。

## 安装依赖

```python
!nvidia-smi
!pip uninstall -y torchao
!pip install -q "peft>=0.14" "accelerate>=1.0"
```

Kaggle images can contain an old optional `torchao` package. Recent PEFT
versions reject `torchao<0.16.0` before LoRA is attached. Standard LoRA in this
project does not use torchao, so it is safe to uninstall it in the notebook.

不要重新安装 PyTorch，也不要直接执行 `pip install -r requirements.txt`，避免覆盖 Kaggle 镜像中的 CUDA PyTorch。

## 定位启动脚本

如果代码 Dataset 已自动解压，运行：

```python
import glob

matches = glob.glob(
    "/kaggle/input/**/run_deberta_v2_xxlarge_lora.sh",
    recursive=True,
)

assert matches, "请先在右侧添加代码 Dataset"
SCRIPT_PATH = matches[0]
print(SCRIPT_PATH)
```

## 启动标准 LoRA 训练

```python
!DATA_ROOT=/kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data \\
 PYTHON_BIN=python \\
 bash {SCRIPT_PATH}
```

默认训练配置：

```text
model = microsoft/deberta-v2-xxlarge
weights = FP32 base weights with FP16 autocast
learning rate = 2e-5
micro batch = 1
gradient accumulation = 16
effective batch = 16
max length = 128
epochs = 3
NEGATIVE_OTHER loss weight = 1.0
R-Drop symmetric KL weight = 0.0
SCL weight = 0.05
SCL temperature = 0.07
SCL queue size = 512
SCL per-class queue size = 128
SCL initial samples per class = 32
LoRA r = 16
LoRA alpha = 32
LoRA targets = query_proj,value_proj
gradient checkpointing = enabled
```

标准 LoRA 不使用 4-bit 加载。

## 输出位置

```text
/kaggle/working/dico_nli_runs/deberta_v2_xxlarge_lora_track1_sclA005_balanced
```

每个 epoch 的 adapter 会保存到：

```text
checkpoints/epoch_1/
checkpoints/epoch_2/
checkpoints/epoch_3/
```

最终 adapter 会保存到输出目录根部。

完整训练参数和环境版本会写入：

```text
training_config.json
```

dev 预测和官方评分位于：

```text
predictions_dev_epoch1.csv
predictions_dev_epoch1_official_scores/
...
```

## 如果显存不足

Kaggle T4/P100 只有 16 GB。标准 LoRA 应优先尝试，不要一开始就量化。

如果标准 LoRA OOM，依次尝试：

```text
MAX_LENGTH=64
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

仍然 OOM 时，在 `main.py` 命令中加入：

```text
--load_in_4bit
```

这会切换到 QLoRA，但默认路径是标准 LoRA。

## 第二阶段：Reversal Consistency

第一阶段完成后，将最佳 adapter 设置为 `STAGE1_ADAPTER`：

```bash
STAGE1_ADAPTER=/kaggle/working/dico_nli_runs/deberta_v2_xxlarge_lora_track1_sclA005_balanced_fp32/checkpoints/epoch_3 \
DATA_ROOT=/kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data \
bash starter_kit/run_deberta_v2_xxlarge_pair_consistency.sh
```

第二阶段默认参数：

```text
pair consistency weight = 0.1
SCL = disabled
R-Drop = disabled
learning rate = 1e-5
epochs = 1
```

第二阶段会把同一 `reverse_pair_id` 的两条序列组成一个训练 unit。一致性项采用
有界 Jensen-Shannon divergence，只作用于三个可逆标签。

## ModernBERT-large 两阶段实验

Stage 1：

```bash
DATA_ROOT=/kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data \
bash starter_kit/run_modernbert_large_scl.sh
```

Stage 2：

```bash
STAGE1_ADAPTER=/kaggle/working/dico_nli_runs/modernbert_large_scl_balanced/checkpoints/epoch_3 \
DATA_ROOT=/kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data \
bash starter_kit/run_modernbert_large_pair_consistency.sh
```

ModernBERT-large 使用 LoRA targets `Wqkv,Wo`，与 DeBERTa 的
`query_proj,value_proj` 不同。

## ModernBERT-large 全参数微调

```bash
DATA_ROOT=/kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data \
bash starter_kit/run_modernbert_large_full.sh
```

默认配置：

```text
full fine-tuning
FP16 AMP + GradScaler
gradient checkpointing
micro batch = 1
gradient accumulation = 16
max length = 128
learning rate = 2e-5
5 epochs
```

每个 epoch 的完整模型保存到：

```text
/kaggle/working/dico_nli_runs/modernbert_large_full_5ep/checkpoints/
```

完整 checkpoint 较大，每个约 1.6 GB。

## NEGATIVE logit 校准

对 ModernBERT-large epoch 4：

```bash
python starter_kit/evaluate_bias_calibration.py \
  --model_path /kaggle/working/dico_nli_runs/modernbert_large_full_5ep/checkpoints/epoch_4 \
  --dev_path /kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data/dev/dico_nli_dev_track1_participant_labeled.csv \
  --reference_path /kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data/dev/dico_nli_dev_track1_reference.csv \
  --output_dir /kaggle/working/modernbert_large_epoch4_bias_calibration \
  --max_length 128 \
  --batch_size 4 \
  --compute_dtype float16 \
  --biases="-1.0,-0.8,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2"
```

## ModernBERT-large seed 43 and 44

```bash
DATA_ROOT=/kaggle/input/datasets/gongbaoxin/dico-nli-dataset/final_data \
bash starter_kit/run_modernbert_large_full_seeds43_44.sh
```

该脚本顺序运行 seed 43 和 seed 44，配置与 E13 的 5-epoch
ModernBERT-large 全参数实验一致。为了控制输出大小，每个 seed 只保存最终模型：

```text
/kaggle/working/dico_nli_runs/modernbert_large_full_seeds43_44/seed_43
/kaggle/working/dico_nli_runs/modernbert_large_full_seeds43_44/seed_44
```
