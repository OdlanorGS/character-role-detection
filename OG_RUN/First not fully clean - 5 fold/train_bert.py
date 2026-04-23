"""
train_bert.py — Training & Evaluation for BERT Role Classifier

Runs stratified K-fold cross-validation on the chunked character data.
Reports per-fold and aggregate F1 scores, plus error analysis.

Usage:
    python train_bert.py

Requires:
    - outputs/chunked_characters.json  (from Step 2.1)
    - dataset.py, model_bert.py        (in same directory)

Outputs:
    - outputs/bert/fold_N/             (model checkpoints per fold)
    - outputs/bert/results.json        (aggregate metrics)
    - outputs/bert/label_encoder.json  (label mapping for inference)
    - Console: per-fold F1, confusion matrices, misclassified characters

pip install required packages:
    torch transformers scikit-learn pandas numpy tqdm

    python -m pip install torch transformers scikit-learn pandas numpy tqdm
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

# Local modules
from dataset import (
    CharacterRoleDataset,
    RoleLabelEncoder,
    load_chunked_data,
    prepare_datasets,
)
from model_bert import (
    BertRoleClassifier,
    WeightedTrainer,
    compute_class_weights,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class TrainConfig:
    # --- Paths ---
    chunks_json: str = "outputs/chunked_characters.json"
    output_dir: str = "outputs/bert"

    # --- Model ---
    model_name: str = "bert-base-uncased"
    max_length: int = 512

    # --- Training ---
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1          # 10% of steps for LR warmup
    use_class_weights: bool = True      # handle role imbalance

    # --- Cross-validation ---
    n_folds: int = 5
    seed: int = 42


CFG = TrainConfig()


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(eval_pred):
    """
    Compute F1 scores for HuggingFace Trainer.
    Called automatically at the end of each evaluation epoch.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro":    f1_score(labels, preds, average="macro"),
        "f1_micro":    f1_score(labels, preds, average="micro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def run_error_analysis(
    trainer,
    eval_dataset: CharacterRoleDataset,
    label_encoder: RoleLabelEncoder,
    fold: int,
) -> dict:
    """
    Detailed error analysis for one fold:
    - Full classification report (precision, recall, F1 per class)
    - Confusion matrix
    - List of misclassified characters with their novels
    """
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    label_names = label_encoder.label_names

    # Classification report
    report = classification_report(
        labels, preds, target_names=label_names, zero_division=0
    )
    print(f"\n--- Fold {fold} Classification Report ---")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(f"Confusion Matrix:\n{cm_df}\n")

    # Misclassified characters
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
    """
    Stratified K-Fold cross-validation.

    Why K-Fold for this project:
      - Only ~30 novels, so a single 80/20 split would leave very
        few validation examples per class.
      - Stratified split ensures each fold preserves the role distribution.
      - Every character gets used for both training and validation
        across folds, giving more reliable F1 estimates.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Load data & fit labels ---
    data, label_encoder = prepare_datasets(cfg.chunks_json, tokenizer=None)
    label_encoder.save(f"{cfg.output_dir}/label_encoder.json")

    # Extract labels for stratification
    all_labels = label_encoder.transform([d["role"] for d in data])
    data_array = np.array(data, dtype=object)

    # --- K-Fold setup ---
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

        # --- Fresh model each fold (no weight leakage) ---
        classifier = BertRoleClassifier(
            model_name=cfg.model_name,
            num_labels=label_encoder.num_labels,
        )
        tokenizer = classifier.get_tokenizer()
        model = classifier.get_model()

        # --- Build datasets ---
        train_dataset = CharacterRoleDataset(
            train_data, tokenizer, label_encoder, cfg.max_length
        )
        val_dataset = CharacterRoleDataset(
            val_data, tokenizer, label_encoder, cfg.max_length
        )

        # --- Class weights (computed from training fold only) ---
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
            report_to="none",          # disable wandb/tensorboard
            fp16=torch.cuda.is_available(),  # mixed precision if GPU
        )

        # --- Train ---
        trainer = WeightedTrainer(
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
    print(f"  AGGREGATE RESULTS — BERT ({cfg.n_folds}-Fold CV)")
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