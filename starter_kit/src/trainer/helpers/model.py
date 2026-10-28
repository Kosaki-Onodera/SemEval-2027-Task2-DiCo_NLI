from typing import Union, Dict, Any, List, Iterable
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import numpy as np


class Model:
    def __init__(self, params: Dict[str, Union[str, int, float, bool]]) -> None:
        self.use_lora: bool = bool(params.get("use_lora", False))
        self.load_in_4bit: bool = bool(params.get("load_in_4bit", False))
        self.gradient_checkpointing: bool = bool(params.get("gradient_checkpointing", False))
        self.compute_dtype = self._resolve_compute_dtype(str(params.get("compute_dtype", "float16")))

        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=str(params["model_path"])
        )

        if params["dropout"] is not None:
            # https://huggingface.co/docs/transformers/model_doc/modernbert#transformers.ModernBertConfig.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Config.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartConfig.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaConfig.classifier_dropout
            # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig.classifier_dropout
            if hasattr(self.config, "classifier_dropout"):
                self.config.classifier_dropout = float(params["dropout"])
            # https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1031
            elif hasattr(self.config, "pooler_dropout") and not hasattr(self.config, "cls_dropout"):
                self.config.cls_dropout = float(params["dropout"])
                self.config.problem_type = "single_label_classification"
            # https://huggingface.co/docs/transformers/model_doc/albert#transformers.AlbertConfig.classifier_dropout_prob
            elif hasattr(self.config, "classifier_dropout_prob"):
                self.config.classifier_dropout_prob = float(params["dropout"])
            else:
                # https://discuss.huggingface.co/t/classifier-dropout-for-decodermodel-forsequenceclassification-classes/113989
                # https://huggingface.co/docs/transformers/model_doc/gpt_neox#transformers.GPTNeoXConfig.classifier_dropout
                # GPT2, OPT
                print("Warning: classifier_dropout attribute not found in AutoConfig. Dropout will make no effect.")

        self.config.num_labels = int(params["num_labels"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=str(params["tokenizer_path"])
        )

        model_kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": str(params["model_path"]),
            "config": self.config,
        }
        if self.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit QLoRA requires a CUDA device.")
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = self.compute_dtype
        elif self.use_lora:
            model_kwargs["torch_dtype"] = self.compute_dtype

        self.model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)

        # For pythia
        if not self.tokenizer.pad_token:
            print("No padding token detected. Expanding tokenizer and setting padding token to [PAD].")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.use_lora:
            self.model = self._attach_lora(params)

        self.emb_size: int = self.model.config.hidden_size
        # self.model.config.max_position_embeddings
        self.max_length: int = int(params["max_length"])

    def _attach_lora(self, params: Dict[str, Union[str, int, float, bool]]):
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

        if self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.gradient_checkpointing,
            )

        adapter_path = params.get("adapter_path")
        if adapter_path:
            print(f"Loading existing LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
                is_trainable=bool(params.get("do_train", False)),
            )
        else:
            target_modules = [
                module.strip()
                for module in str(params.get("lora_target_modules", "query_proj,value_proj")).split(",")
                if module.strip()
            ]
            if not target_modules:
                raise ValueError("--lora_target_modules must contain at least one module name.")

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=int(params.get("lora_r", 16)),
                lora_alpha=int(params.get("lora_alpha", 32)),
                lora_dropout=float(params.get("lora_dropout", 0.05)),
                target_modules=target_modules,
                modules_to_save=["classifier"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

        if self.gradient_checkpointing:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()

        self.model.print_trainable_parameters()
        return self.model

    @staticmethod
    def _resolve_compute_dtype(name: str) -> torch.dtype:
        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if name not in dtypes:
            raise ValueError(f"Unsupported compute dtype: {name}")
        return dtypes[name]

    def forward(self, batch: Dict[str, torch.Tensor], device: torch.device) -> SequenceClassifierOutput:
        inputs: Dict[str, torch.Tensor] = {
            "input_ids": batch["input_ids"].to(device=device),
            "attention_mask": batch["attention_mask"].to(device=device),
        }
        
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device=device)

        # Check if labels are provided in the batch
        if "labels" in batch:
            inputs["labels"] = batch["labels"].to(device=device)

        return self.model(**inputs)

    def tokenize(self, data: np.ndarray) -> dict[str, Any]:
        texts_first, texts_second = zip(*data)
        texts_first: np.ndarray
        texts_second: np.ndarray

        separator: str = self.tokenizer.sep_token or "[SEP]"
        combined_texts: List[str] = [
            f"{first} {separator} {second}"
            for first, second in zip(texts_first, texts_second)
        ]

        encs: BatchEncoding = self.tokenizer(
            combined_texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        return encs.data
    
    def to(self, device: torch.device) -> None:
        if not self.load_in_4bit:
            self.model.to(device=device)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def decode(self, token_ids: List[List[int]]) -> List[str]:
        decoded_outputs:List[str] = self.tokenizer.batch_decode(sequences=token_ids, skip_special_tokens=False)
        return [text.replace(self.tokenizer.pad_token, "") for text in decoded_outputs]

    def save(self, path: str) -> None:
        self.model.save_pretrained(save_directory=path)
        self.tokenizer.save_pretrained(save_directory=path)
