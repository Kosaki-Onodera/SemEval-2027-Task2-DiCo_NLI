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
precision = float16
micro batch = 1
gradient accumulation = 16
effective batch = 16
max length = 128
epochs = 3
LoRA r = 16
LoRA alpha = 32
LoRA targets = query_proj,value_proj
gradient checkpointing = enabled
```

标准 LoRA 不使用 4-bit 加载。

## 输出位置

```text
/kaggle/working/dico_nli_runs/deberta_v2_xxlarge_lora_track1
```

每个 epoch 的 adapter 会保存到：

```text
checkpoints/epoch_1/
checkpoints/epoch_2/
checkpoints/epoch_3/
```

最终 adapter 会保存到输出目录根部。

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
