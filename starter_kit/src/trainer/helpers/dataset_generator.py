from torch.utils.data import Dataset
import numpy as np
import torch
from typing import List, Dict, Optional, Any

class NLIDataset(Dataset):
    def __init__(
        self, 
        X: Dict[str, np.ndarray], 
        ids: np.ndarray,
        device: torch.device, 
        labels: Optional[np.ndarray] = None 
    ) -> None:
        super().__init__()
        self.device: torch.device = device
        self.X: Dict[str, np.ndarray] = X
        self.ids: np.ndarray = ids
        
        self.labels: Optional[np.ndarray] = None
        if labels is not None:
            self.labels = labels.reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.X["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "id": str(self.ids[idx]),  # Cast to string to ensure native Python type
            "input_ids": torch.clone(input=self.X["input_ids"][idx]),
            "attention_mask": torch.clone(input=self.X["attention_mask"][idx]),
        }

        # Add labels only if they were provided
        if self.labels is not None:
            # LabelEncoder returns np.int_ which is int32 on Windows; cross-entropy needs int64.
            item["labels"] = torch.from_numpy(self.labels[idx]).long()

        # Add token_type_ids if available
        if "token_type_ids" in self.X:
            item["token_type_ids"] = torch.clone(input=self.X["token_type_ids"][idx])
            
        return item

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated: Dict[str, Any] = {
            "ids": [item["id"] for item in batch],
            "input_ids": torch.stack(tensors=[item["input_ids"] for item in batch]).to(device=self.device),
            "attention_mask": torch.stack(tensors=[item["attention_mask"] for item in batch]).to(device=self.device),
        }
        
        if "labels" in batch[0]:
            collated["labels"] = torch.stack(tensors=[item["labels"] for item in batch]).to(device=self.device)

        if "token_type_ids" in batch[0]:
            collated["token_type_ids"] = torch.stack(tensors=[item["token_type_ids"] for item in batch]).to(device=self.device)

        return collated


class ReversalPairDataset(Dataset):
    """Dataset whose items contain one reversal pair or one negative instance."""

    def __init__(
        self,
        X: Dict[str, torch.Tensor],
        ids: np.ndarray,
        labels: np.ndarray,
        groups: List[Dict[str, Any]],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.X = X
        self.ids = ids
        self.labels = labels
        self.groups = groups
        self.device = device

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        group = self.groups[idx]
        row_indices = group["row_indices"]
        item: Dict[str, Any] = {
            "ids": [str(self.ids[row_index]) for row_index in row_indices],
            "input_ids": self.X["input_ids"][row_indices].clone(),
            "attention_mask": self.X["attention_mask"][row_indices].clone(),
            "labels": torch.from_numpy(
                self.labels[row_indices].reshape(-1, 1)
            ).long(),
            "is_reversal_pair": bool(group["is_reversal_pair"]),
        }
        if "token_type_ids" in self.X:
            item["token_type_ids"] = self.X["token_type_ids"][row_indices].clone()
        return item

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        attention_mask = []
        labels = []
        token_type_ids = []
        ids: List[str] = []
        groups: List[Dict[str, Any]] = []
        offset = 0

        for item in batch:
            count = int(item["input_ids"].shape[0])
            groups.append(
                {
                    "start": offset,
                    "end": offset + count,
                    "is_reversal_pair": item["is_reversal_pair"],
                }
            )
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
            labels.append(item["labels"])
            ids.extend(item["ids"])
            if "token_type_ids" in item:
                token_type_ids.append(item["token_type_ids"])
            offset += count

        collated: Dict[str, Any] = {
            "ids": ids,
            "input_ids": torch.cat(input_ids, dim=0).to(device=self.device),
            "attention_mask": torch.cat(attention_mask, dim=0).to(device=self.device),
            "labels": torch.cat(labels, dim=0).to(device=self.device),
            "pair_groups": groups,
        }
        if token_type_ids:
            collated["token_type_ids"] = torch.cat(
                token_type_ids,
                dim=0,
            ).to(device=self.device)
        return collated
