"""
train_longformer.py — Training & Evaluation for Longformer Role Classifier

Same K-fold structure as train_bert.py, but with:
  - 4096 token max length (vs BERT's 512)
  - Global attention on [CLS] for full-sequence classification
  - Smaller batch size (Longformer uses more memory per sample)
  - Lower learning rate (Longformer is more sensitive to LR)

Usage:
    python train_longformer.py

Requires:
    - outputs/chunked_characters.json  (from Step 2.1)
    - dataset.py, model_longformer.py  (in same directory)

Outputs:
    - outputs/longformer/fold_N/            (model checkpoints per fold)
    - outputs/longformer/results.json       (aggregate metrics)
    - outputs/longformer/label_encoder.json (label mapping)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import TrainingArguments

# Local modules (reuse dataset.py from Step 2.2)
from dataset import (
    CharacterRoleDataset,
    RoleLabelEncoder,
    load_chunked_data,
    prepare_datasets,
)
from model_longformer import (
    LongformerRoleClassifier,
    LongformerWeightedTrainer,
)
from model_bert import compute_class_weights  # reuse the same function


# =============================================================================
# CONFIGURATION
# =============================================================================

class TrainConfig:
    # --- Paths ---
    chunks_json: str = "outputs/chunked_characters.json"
    output_dir: str = "outputs/longformer"

    # --- Model ---
    model_name: str = "allenai/longformer-base-"
    max_length: int = 4096              # 8x BERT's limit

    # --- Training ---
    # Longformer needs gentler hyperparameters than BERT:
    #   - Smaller batch size (more memory per sample due to longer sequences)
    #   - Lower learning rate (Longformer is pretrained differently)
    epochs: int = 10
    batch_size: int = 2                 # reduced from BERT's 8 — 4096 tokens per sample
    gradient_accumulation: int = 4      # effective batch size = 2 * 4 = 8
    learning_rate: float = 1e-5         # half of BERT's 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    use_class_weights: bool = True

    # --- Cross-validation ---
    n_folds: int = 5
    seed: int = 42


CFG = TrainConfig()


# =============================================================================
# METRICS (same as BERT)
# =============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro":    f1_score(labels, preds, average="macro"),
        "f1_micro":    f1_score(labels, preds, average="micro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# =============================================================================
# ERROR ANALYSIS (same as BERT)
# =============================================================================

def run_error_analysis(
    trainer,
    eval_dataset: CharacterRoleDataset,
    label_encoder: RoleLabelEncoder,
    fold: int,
) -> dict:
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    label_names = label_encoder.label_names

    report = classification_report(
        labels, preds, target_names=label_names, zero_division=0
    )
    print(f"\n--- Fold {fold} Classification Report ---")
    print(report)

    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(f"Confusion Matrix:\n{cm_df}\n")

    misclassified = []
    for i in range(len(preds)):
        if preds[i] != labels[i]:
            meta = eval_dataset.get_metadata(i)
            misclassified.append({
                "character": meta["character"],
                "novel": meta["novel"],
                "true_role": label_names[labels[i]],
                "predicted": label_names[preds[i]],
            })

    if misclassified:
        print(f"Misclassified ({len(misclassified)}):")
        for m in misclassified:
            print(f"  {m['character']:<35} ({m['novel']}): "
                  f"true={m['true_role']}, pred={m['predicted']}")

    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "misclassified": misclassified,
    }


# =============================================================================
# K-FOLD TRAINING
# =============================================================================

def train_kfold(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Load data & fit labels ---
    data, label_encoder = prepare_datasets(cfg.chunks_json, tokenizer=None)
    label_encoder.save(f"{cfg.output_dir}/label_encoder.json")

    all_labels = label_encoder.transform([d["role"] for d in data])
    data_array = np.array(data, dtype=object)

    skf = StratifiedKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
    )
    fold_metrics = []
    all_error_analyses = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(data_array, all_labels), start=1
    ):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{cfg.n_folds}")
        print(f"{'='*60}")

        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        print(f"  Train: {len(train_data)} characters")
        print(f"  Val:   {len(val_data)} characters")

        # --- Fresh model each fold ---
        classifier = LongformerRoleClassifier(
            model_name=cfg.model_name,
            num_labels=label_encoder.num_labels,
        )
        tokenizer = classifier.get_tokenizer()
        model = classifier.get_model()

        # --- Build datasets (same class, different max_length) ---
        train_dataset = CharacterRoleDataset(
            train_data, tokenizer, label_encoder, cfg.max_length
        )
        val_dataset = CharacterRoleDataset(
            val_data, tokenizer, label_encoder, cfg.max_length
        )

        # --- Class weights ---
        class_weights_tensor = None
        if cfg.use_class_weights:
            train_labels = [s["label"] for s in train_dataset.samples]
            weights = compute_class_weights(
                train_labels, label_encoder.num_labels
            )
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

        # --- Training arguments ---
        fold_output = f"{cfg.output_dir}/fold_{fold}"
        training_args = TrainingArguments(
            output_dir=fold_output,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            seed=cfg.seed,
            logging_steps=10,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        # --- Train with Longformer-specific trainer ---
        trainer = LongformerWeightedTrainer(
            class_weights=class_weights_tensor,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # --- Evaluate ---
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)
        print(f"\nFold {fold} Results:")
        print(f"  F1 Macro:    {metrics['eval_f1_macro']:.4f}")
        print(f"  F1 Micro:    {metrics['eval_f1_micro']:.4f}")
        print(f"  F1 Weighted: {metrics['eval_f1_weighted']:.4f}")

        # --- Error analysis ---
        error_info = run_error_analysis(
            trainer, val_dataset, label_encoder, fold
        )
        all_error_analyses.append(error_info)

    # =================================================================
    # AGGREGATE RESULTS
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS — Longformer ({cfg.n_folds}-Fold CV)")
    print(f"{'='*60}")

    summary = {}
    for key in ["eval_f1_macro", "eval_f1_micro", "eval_f1_weighted"]:
        values = [m[key] for m in fold_metrics]
        mean = np.mean(values)
        std = np.std(values)
        summary[key] = {"mean": mean, "std": std, "per_fold": values}
        print(f"  {key}: {mean:.4f} ± {std:.4f}")

    # Save results
    results = {
        "model": cfg.model_name,
        "n_folds": cfg.n_folds,
        "epochs": cfg.epochs,
        "max_length": cfg.max_length,
        "summary": {
            k: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)}
            for k, v in summary.items()
        },
        "per_fold": fold_metrics,
        "error_analyses": all_error_analyses,
    }

    results_path = f"{cfg.output_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved → {results_path}")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = train_kfold(CFG)