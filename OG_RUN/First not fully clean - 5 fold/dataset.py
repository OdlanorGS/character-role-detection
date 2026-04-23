"""
dataset.py — Character Role Dataset for BERT

Loads chunked_characters.json from Step 2.1 and prepares
tokenized inputs for a standard transformer (512 token limit).

Chunk combination strategy:
    Each character has 3 chunks (first, last, frequent mention).
    We concatenate them with special markers so BERT sees a compressed
    view of the character's arc in a single 512-token sequence:

    [CLS] Character: {name} [SEP] First: {chunk} [SEP] Last: {chunk} [SEP] Key: {chunk} [SEP]
"""

import json
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional
from pathlib import Path # for file path handling


# =============================================================================
# LABEL ENCODING
# =============================================================================

class RoleLabelEncoder:
    """
    Wraps sklearn's LabelEncoder with save/load and label-name lookups.
    Keeps label mapping consistent across training and evaluation.
    """

    def __init__(self):
        self.encoder = LabelEncoder() # sklearn encoder to convert role labels to integers
        self.is_fitted = False

    def fit(self, roles: List[str]) -> "RoleLabelEncoder": # Fit the encoder to the list of role labels
        self.encoder.fit(roles)
        self.is_fitted = True
        return self

    def transform(self, roles: List[str]) -> List[int]: # Convert role labels to integer IDs
        return self.encoder.transform(roles).tolist()

    def inverse_transform(self, ids: List[int]) -> List[str]: # Convert integer IDs back to role labels
        return self.encoder.inverse_transform(ids).tolist()

    @property
    def num_labels(self) -> int: # Return the number of unique role labels (classes)
        return len(self.encoder.classes_)

    @property
    def label_names(self) -> List[str]: # Return the list of role labels in the order of their integer IDs
        return list(self.encoder.classes_)

    def save(self, path: str):
        """Save label mapping as JSON for reproducibility."""
        mapping = {
            "classes": list(self.encoder.classes_),
            "label_to_id": {
                label: int(idx)
                for idx, label in enumerate(self.encoder.classes_)
            },
        }
        with open(path, "w") as f:
            json.dump(mapping, f, indent=2)

    def load(self, path: str) -> "RoleLabelEncoder":
        """Load a previously saved label mapping."""
        with open(path) as f:
            mapping = json.load(f)
        self.encoder.classes_ = mapping["classes"]
        self.is_fitted = True
        return self


# =============================================================================
# DATASET
# =============================================================================

class CharacterRoleDataset(Dataset):
    """
    PyTorch Dataset that tokenizes character chunks for BERT.

    Each sample concatenates the three chunks (first, last, frequent)
    into a single string, then tokenizes to max_length tokens.

    Args:
        data:        List of character dicts from chunked_characters.json
        tokenizer:   HuggingFace tokenizer (e.g., BertTokenizer)
        label_encoder: Fitted RoleLabelEncoder
        max_length:  Max token length (512 for BERT)
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        label_encoder: RoleLabelEncoder,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.samples = []

        for entry in data:
            # Build the combined input string from chunks
            text = self._build_input_text(entry)
            if not text:
                continue

            self.samples.append({
                "text": text,
                "label": label_encoder.transform([entry["role"]])[0],
                "novel": entry["novel"],
                "character": entry["character"],
                "role": entry["role"],
            })

    def _build_input_text(self, entry: Dict) -> Optional[str]:
        """
        Combine the three chunks into a single input string.

        Format:
            Character: {name} [SEP]
            First appearance: {first_chunk} [SEP]
            Last appearance: {last_chunk} [SEP]
            Key scene: {frequent_chunk}
        """
        chunks = entry.get("chunks", {})
        if not chunks:
            return None

        first = chunks.get("first", {}).get("text", "")
        last = chunks.get("last", {}).get("text", "")
        frequent = chunks.get("frequent", {}).get("text", "")

        if not any([first, last, frequent]):
            return None

        parts = [f"Character: {entry['character']}"]
        if first:
            parts.append(f"First appearance: {first}")
        if last:
            parts.append(f"Last appearance: {last}")
        if frequent:
            parts.append(f"Key scene: {frequent}")

        return " [SEP] ".join(parts)

    def __len__(self) -> int: # Return the number of samples in the dataset
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: # Return tokenized input and label for the sample at index idx
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0), # remove batch dimension
            "attention_mask": encoding["attention_mask"].squeeze(0), # remove batch dimension
            "labels": torch.tensor(sample["label"], dtype=torch.long), # role label as integer
        }

    def get_metadata(self, idx: int) -> Dict:
        """Return novel/character/role info for error analysis."""
        return {
            k: self.samples[idx][k]
            for k in ("novel", "character", "role")
        }


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_chunked_data(json_path: str) -> List[Dict]:
    """Load the chunked_characters.json from Step 2.1."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter to only successfully chunked characters
    valid = [d for d in data if d["status"] == "ok"]
    skipped = len(data) - len(valid)

    print(f"Loaded {len(valid)} characters ({skipped} skipped — no chunks)")
    return valid


def prepare_datasets(
    json_path: str,
    tokenizer,
    max_length: int = 512,
    label_encoder: Optional[RoleLabelEncoder] = None,
) -> Tuple[List[Dict], RoleLabelEncoder]:
    """
    Load data and fit label encoder. Returns the raw data list
    and fitted encoder — the actual Dataset objects are created
    per-fold in the training script.
    """
    data = load_chunked_data(json_path)

    if label_encoder is None:
        label_encoder = RoleLabelEncoder()
        roles = [d["role"] for d in data]
        label_encoder.fit(roles)

    print(f"Labels: {label_encoder.label_names}")
    print(f"Label distribution:")
    from collections import Counter
    role_counts = Counter(d["role"] for d in data)
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")

    return data, label_encoder