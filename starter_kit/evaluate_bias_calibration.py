import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


STARTER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STARTER_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_functions import score_files
from src.trainer.helpers.data_reader import label_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a NEGATIVE_OTHER logit bias on a full DiCo-NLI model."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dev_path", required=True)
    parser.add_argument("--reference_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--compute_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--biases",
        default="-1.0,-0.8,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.compute_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=4,
        torch_dtype=dtype,
        device_map={"": 0} if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    dev = pd.read_csv(args.dev_path)
    required = {"instance_id", "text1", "text2"}
    missing = required - set(dev.columns)
    if missing:
        raise ValueError(f"Missing dev columns: {', '.join(sorted(missing))}")

    separator = tokenizer.sep_token or "[SEP]"
    texts = [
        f"{clean_text(text1)} {separator} {clean_text(text2)}"
        for text1, text2 in zip(dev["text1"], dev["text2"])
    ]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    all_logits = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = {
                name: tensor[start:start + args.batch_size].to(device)
                for name, tensor in encoded.items()
            }
            logits = model(**batch).logits
            all_logits.append(logits.float().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    np.save(output_dir / "dev_logits.npy", logits)

    classes = [str(label) for label in label_encoder.classes_]
    negative_index = classes.index("NEGATIVE_OTHER")
    biases = [float(value) for value in args.biases.split(",")]
    results = []

    for bias in biases:
        adjusted_logits = logits.copy()
        adjusted_logits[:, negative_index] += bias
        predictions = np.argmax(adjusted_logits, axis=1)
        predicted_labels = [classes[index] for index in predictions]

        tag = f"{bias:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
        prediction_path = output_dir / f"predictions_bias_{tag}.csv"
        pd.DataFrame(
            {
                "instance_id": dev["instance_id"],
                "label": predicted_labels,
            }
        ).to_csv(prediction_path, index=False)

        report = score_files(
            args.reference_path,
            prediction_path,
            write_outputs=False,
        )
        payload = report.to_dict()
        negative = payload["classification"]["per_label"]["NEGATIVE_OTHER"]
        results.append(
            {
                "bias": bias,
                "weighted_f1": payload["weighted_f1"],
                "soft_cons": payload["soft_cons"],
                "hard_cons": payload["hard_cons"],
                "accuracy": payload["classification"]["accuracy"],
                "negative_precision": negative["precision"],
                "negative_recall": negative["recall"],
                "negative_f1": negative["f1"],
                "negative_predicted": negative["predicted"],
                "prediction_path": str(prediction_path),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "bias_results.csv", index=False)
    (output_dir / "bias_results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    baseline = min(results, key=lambda row: abs(row["bias"]))
    dominated = [
        row
        for row in results
        if (
            abs(row["bias"] - baseline["bias"]) > 1e-9
            and
            row["weighted_f1"] >= baseline["weighted_f1"]
            and row["soft_cons"] >= baseline["soft_cons"]
            and row["hard_cons"] >= baseline["hard_cons"]
        )
    ]
    if dominated:
        best = max(dominated, key=lambda row: row["weighted_f1"])
        print("Found a bias that is not worse on any official metric.")
    else:
        best = max(results, key=lambda row: row["weighted_f1"])
        print("No metric-dominating bias found; selected maximum Weighted F1.")

    score_files(
        args.reference_path,
        best["prediction_path"],
        output_dir=output_dir / "best_scores",
    )

    print(results_df.to_string(index=False))
    print("\nSelected candidate:")
    for key, value in best.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
