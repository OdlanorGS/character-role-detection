"""
dataset.py — Character Role Dataset for Transformers

Loads chunked_characters.json from Step 2.1 and prepares
tokenized inputs for transformer classifiers.

Chunk combination strategy:
    Each character has 3 chunks (first, last, frequent mention).
    We concatenate them with semantic labels so the model sees a
    compressed view of the character's arc in a single sequence:

    [CLS] Character: {name} [SEP] First: {chunk} [SEP] Last: {chunk} [SEP] Key: {chunk} [SEP]

Performance note:
    Tokenization happens ONCE in __init__, not per-batch in __getitem__.
    On CPU this avoids re-running the tokenizer every epoch × every fold,
    which was the main hidden slowdown in the previous version.

Truncation strategy:
    When max_length is small (e.g., 512 for BERT), the concatenated chunks
    will be truncated. To ensure the model sees the most useful parts of
    ALL three chunks rather than just the first chunk in full, we
    budget tokens equally across the three chunks before concatenating.
"""

import json
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# =============================================================================
# LABEL ENCODING
# =============================================================================

class RoleLabelEncoder:
    """
    Wraps sklearn's LabelEncoder with save/load and label-name lookups.
    Keeps label mapping consistent across training and evaluation.
    """

    def __init__(self):
        self.encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, roles: List[str]) -> "RoleLabelEncoder":
        self.encoder.fit(roles)
        self.is_fitted = True
        return self

    def transform(self, roles: List[str]) -> List[int]:
        return self.encoder.transform(roles).tolist()

    def inverse_transform(self, ids: List[int]) -> List[str]:
        return self.encoder.inverse_transform(ids).tolist()

    @property
    def num_labels(self) -> int:
        return len(self.encoder.classes_)

    @property
    def label_names(self) -> List[str]:
        return list(self.encoder.classes_)

    def save(self, path: str):
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
        with open(path) as f:
            mapping = json.load(f)
        self.encoder.classes_ = mapping["classes"]
        self.is_fitted = True
        return self


# =============================================================================
# SMART CHUNK BUDGETING
# =============================================================================

def budget_chunks(
    chunks: Dict[str, str],
    character_name: str,
    tokenizer,
    max_length: int,
) -> str:
    """
    Allocate tokens fairly across the three chunks so truncation
    doesn't eat the last and frequent chunks entirely.

    Problem with naive concatenation:
        first_chunk (800 tokens) + last_chunk (600 tokens) + frequent (500 tokens)
        → 1900 tokens → BERT truncates to 512 → model only sees first chunk

    Solution:
        Reserve tokens for the character name and separators (~20 tokens),
        then split remaining budget equally across the three chunks.
        Each chunk gets truncated individually before concatenation.

    For Longformer (max_length=4096), chunks fit without truncation,
    so this function effectively passes everything through unchanged.
    """
    # Reserve tokens for name prefix, separators, special tokens, and section labels
    # [CLS] + [SEP]×3 + "Character: {name}" + "First appearance:" + "Last appearance:" + "Key scene:"
    # Previous value of 30 was too tight — caused 548 > 512 overflow on some samples
    overhead_tokens = 50
    available = max_length - overhead_tokens

    chunk_keys = ["first", "last", "frequent"]
    chunk_texts = {k: chunks.get(k, {}).get("text", "") for k in chunk_keys}

    # Tokenize each chunk to measure actual token lengths
    chunk_token_lengths = {}
    for k, text in chunk_texts.items():
        if text:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            chunk_token_lengths[k] = len(tokens)
        else:
            chunk_token_lengths[k] = 0

    total_tokens = sum(chunk_token_lengths.values())

    # If everything fits, no budgeting needed
    if total_tokens <= available:
        parts = [f"Character: {character_name}"]
        labels = {"first": "First appearance", "last": "Last appearance", "frequent": "Key scene"}
        for k in chunk_keys:
            if chunk_texts[k]:
                parts.append(f"{labels[k]}: {chunk_texts[k]}")
        return " [SEP] ".join(parts)

    # Budget equally, then redistribute unused budget
    per_chunk_budget = available // 3

    truncated = {}
    leftover = 0

    # First pass: truncate chunks that exceed budget, track savings from short chunks
    for k in chunk_keys:
        if chunk_token_lengths[k] <= per_chunk_budget:
            truncated[k] = chunk_texts[k]  # fits as-is
            leftover += per_chunk_budget - chunk_token_lengths[k]
        else:
            truncated[k] = None  # needs truncation, handle in second pass

    # Second pass: distribute leftover to chunks that need it
    needs_truncation = [k for k in chunk_keys if truncated[k] is None]
    extra_per = leftover // max(len(needs_truncation), 1)

    for k in needs_truncation:
        budget = per_chunk_budget + extra_per
        tokens = tokenizer.encode(chunk_texts[k], add_special_tokens=False)
        truncated_tokens = tokens[:budget]
        truncated[k] = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    # Build final string
    parts = [f"Character: {character_name}"]
    labels = {"first": "First appearance", "last": "Last appearance", "frequent": "Key scene"}
    for k in chunk_keys:
        if truncated[k]:
            parts.append(f"{labels[k]}: {truncated[k]}")

    return " [SEP] ".join(parts)


# =============================================================================
# DATASET
# =============================================================================

class CharacterRoleDataset(Dataset):
    """
    PyTorch Dataset that pre-tokenizes character chunks.

    All tokenization happens in __init__ so __getitem__ is just
    a tensor lookup — no repeated tokenizer calls across epochs.

    Args:
        data:          List of character dicts from chunked_characters.json
        tokenizer:     HuggingFace tokenizer (BERT or Longformer)
        label_encoder: Fitted RoleLabelEncoder
        max_length:    Max token length (512 for BERT, 4096 for Longformer)
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
        self.samples = []           # metadata for error analysis
        self.encodings = []         # pre-tokenized tensors

        for entry in data:
            chunks = entry.get("chunks", {})
            if not chunks:
                continue

            # Budget tokens across chunks, then build input string
            text = budget_chunks(
                chunks, entry["character"], tokenizer, max_length
            )

            if not text:
                continue

            # Tokenize ONCE here — not in __getitem__
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(
                    label_encoder.transform([entry["role"]])[0],
                    dtype=torch.long,
                ),
            })

            self.samples.append({
                "text": text,
                "label": label_encoder.transform([entry["role"]])[0],
                "novel": entry["novel"],
                "character": entry["character"],
                "role": entry["role"],
            })

        print(f"  Pre-tokenized {len(self.encodings)} samples "
              f"(max_length={max_length})")

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Pure tensor lookup — no tokenizer overhead."""
        return self.encodings[idx]

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
    and fitted encoder — Dataset objects are created per-fold
    in the training scripts.
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