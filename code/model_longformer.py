"""
model_longformer.py — Longformer Classifier for Character Role Detection
Requires: dataset.py, trainer.py 

Key difference from BERT:
  - Supports up to 4096 tokens (8x BERT's 512 limit)
  - Same chunks from Step 2.1, but we can fit far more context per character

  Can also experiment with larger window_chars in Step 2.1 (e.g., 4000)
  to give Longformer even more context per chunk.
"""

import torch
import torch.nn as nn
from transformers import (
    LongformerTokenizer,
    LongformerForSequenceClassification,
    AutoConfig,
    Trainer,
)
from typing import Optional, List


# =============================================================================
# LONGFORMER CLASSIFIER
# =============================================================================

class LongformerRoleClassifier:
    """
    Longformer for sequence classification.

    Architecture:
      - Attention is local (sliding window) by default, which scales
        linearly with sequence length instead of quadratically.
      - The [CLS] token gets global attentionon.
      - Max position embeddings = 4096 tokens.
    """

    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096", # Longformer base with 4096 token limit
        num_labels: int = 5, # Number of character role classes, default 5 for our 5 roles
        class_weights: Optional[List[float]] = None, # same class weights as BERT for weighted loss, but can be tuned separately if needed
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU if available, else CPU

        # Load tokenizer
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name) # Longformer tokenizer

        # Load model with classification head
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model = LongformerForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
        )

        # Store class weights for weighted loss
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights, dtype=torch.float32
            ).to(self.device)

        self.model.to(self.device)
        print(f"Initialized {model_name} with {num_labels} labels on {self.device}") # 

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model


# =============================================================================
# WEIGHTED TRAINER (with global attention on [CLS])
# =============================================================================

class LongformerWeightedTrainer(Trainer):
    """
    Custom Trainer for Longformer that:
      1. Sets global attention on the [CLS] token (position 0)
      2. Applies class weights to the loss function

    Why global attention matters:
      Longformer's default attention is local — each token only attends
      to its neighbors within a sliding window. The [CLS] token is what
      the classification head reads, so it MUST attend to the full
      sequence. Setting global_attention_mask[0] = 1 enables this.

      Without this, [CLS] would only see the first ~512 tokens,
      defeating the purpose of using Longformer.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        # --- Set global attention on [CLS] token ---
        # global_attention_mask: 0 = local attention, 1 = global attention
        # We set position 0 ([CLS]) to global so it sees the full sequence
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1  # [CLS] token at position 0
        inputs["global_attention_mask"] = global_attention_mask

        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss