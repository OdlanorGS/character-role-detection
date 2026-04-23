"""
model_bert.py — BERT Classifier for Character Role Detection

Wraps HuggingFace's BERT with:
  - Sequence classification head (num_labels = number of roles)
  - Optional class-weighted loss to handle imbalanced roles (e.g. more allies than villains)

The 512-token limit is handled at the dataset level, where character mentions and their surrounding context are chunked. 
The model itself just sees these chunks as input.
"""

import torch # for tensor operations and device management
import torch.nn as nn # for defining the loss function
import numpy as np # for computing class weights

from transformers import ( 
    AutoTokenizer, #
    AutoModelForSequenceClassification,
    AutoConfig,
) # HuggingFace transformers for BERT model and tokenizer

from typing import Optional, List # for type annotations


# =============================================================================
# BERT CLASSIFIER
# =============================================================================

class BertRoleClassifier:
    """
    Standard BERT for sequence classification.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased", # pre-trained BERT model to use
        num_labels: int = 5, # will be overwritten by the actual number of roles in the dataset (e.g. 3 or 4)
        class_weights: Optional[List[float]] = None, # optional list of class weights for weighted loss (computed from the training labels to handle class imbalance)
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # automatically use GPU if available, otherwise fall back to CPU

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # HuggingFace tokenizer for the specified BERT model

        # Load model with classification head
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
        )

        # Store class weights for weighted loss
        self.class_weights = None # will be set to a tensor if class_weights are provided, otherwise remains None for unweighted loss
        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights, dtype=torch.float32
            ).to(self.device)

        self.model.to(self.device)
        print(f"Initialized {model_name} with {num_labels} labels on {self.device}")

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model


# =============================================================================
# CLASS WEIGHT COMPUTATION
# =============================================================================

def compute_class_weights(labels: List[int], num_classes: int) -> List[float]:
    """
    Compute inverse-frequency class weights.

    These weights tell the loss function to penalize mistakes on rare classes more heavily.

    Formula: weight_i = total_samples / (num_classes * count_i)

    Example:
        labels = [0,0,0,0,1,1,2]  (4 heroes, 2 villains, 1 neutral)
        weights = [7/(3*4), 7/(3*2), 7/(3*1)] = [0.58, 1.17, 2.33]  neutral mistakes penalized 4x more than hero mistakes 
        because neutral is the rarest class. 
        A model that predicts "hero" for everything would have a high loss on the neutral examples, 
        encouraging it to learn to distinguish them better.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    weights = []
    for count in counts:
        if count == 0:
            weights.append(1.0)  # avoid division by zero
        else:
            weights.append(total / (num_classes * count))

    print(f"Class counts: {counts}")
    print(f"Class weights: {[f'{w:.3f}' for w in weights]}")
    return weights


# =============================================================================
# WEIGHTED TRAINER
# =============================================================================

from transformers import Trainer


class WeightedTrainer(Trainer):
    """
    Custom HuggingFace Trainer that applies class weights to the loss.

    The standard Trainer uses unweighted CrossEntropyLoss, which treats
    all classes equally. With imbalanced roles, this biases the model
    toward predicting the majority class. WeightedTrainer injects
    class weights into the loss computation.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            # Move weights to same device as logits
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss