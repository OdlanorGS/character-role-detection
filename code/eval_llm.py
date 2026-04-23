"""
eval_llm.py — Zero-Shot LLM Evaluation for Character Role Classification

Evaluates Claude, GPT-4o, Gemini, and Ollama on the same character role
classification task as BERT and Longformer, using zero-shot prompting.

Two evaluation conditions:
  --condition fair   Same 4096-token budget as Longformer (controls for input,
                     isolates model capability — direct apples-to-apples comparison)
  --condition full   Uncapped context — LLMs receive the complete raw chunks
                     with no truncation, showing their actual ceiling performance

Both conditions use the same prompt template, same fold splits (n_folds=5,
seed=42), and produce the same results.json schema as BERT/Longformer runs.

Colab note:
  - Claude, GPT, Gemini: run on Colab (need API keys)
  - Ollama: run LOCALLY — Ollama is a local server, won't work on Colab.
    See OLLAMA SETUP section in OllamaEvaluator below.

Usage:
  # Smoke test one model, 1 fold
  python eval_llm.py --models claude --smoke-test

  # Fair comparison (same input budget as Longformer)
  python eval_llm.py --models claude gpt gemini --condition fair

  # Full context (uncapped — LLMs at their best)
  python eval_llm.py --models claude gpt gemini --condition full

  # Ollama locally
  python eval_llm.py --models ollama --ollama-model llama3.2 --condition fair
  python eval_llm.py --models ollama --ollama-model llama3.2 --condition full

Install:
  pip install anthropic openai google-generativeai ollama tqdm transformers
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from dataset import load_chunked_data, RoleLabelEncoder, budget_chunks

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os

for name in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN"]:
    val = os.getenv(name)
    print(f"{name}: {'FOUND' if val else 'MISSING'}")
# =============================================================================
# CONFIGURATION
# =============================================================================

class EvalConfig:
    # --- Paths ---
    chunks_json: str  = "../outputs/chunked_characters.json"

    # Output dir is set dynamically from --condition flag:
    #   outputs/llm_eval_fair/    ← fair condition
    #   outputs/llm_eval_full/    ← full context condition
    output_dir: str   = "outputs/llm_eval_fair"

    # --- Condition ---
    condition: str    = "fair"   # "fair" | "full"

    # fair: same 4096-token budget as Longformer — controls for input
    # full: None = no truncation — LLMs receive complete raw chunks
    fair_max_length: int           = 4096
    full_max_length: Optional[int] = None

    @property
    def active_max_length(self) -> Optional[int]:
        return self.fair_max_length if self.condition == "fair" else self.full_max_length

    # --- Cross-validation ---
    # MUST match BERT/Longformer: same folds + seed → same val splits
    n_folds: int      = 5
    seed: int         = 42

    # --- Rate limiting (requests per minute) ---
    claude_rpm: int   = 50
    gpt_rpm: int      = 60
    gemini_rpm: int   = 60
    ollama_rpm: int   = 999    # local, no limit

    # --- Retry ---
    max_retries: int   = 3
    retry_delay: float = 5.0   # seconds; multiplied by attempt number (backoff)

    # --- Model names ---
    claude_model: str  = "claude-sonnet-4-6"
    gpt_model: str     = "gpt-4o-mini"        # swap to gpt-4o for best perf
    gemini_model: str  = "gemini-2.5-flash"  # swap to gemini-1.5-pro for best
    ollama_model: str  = "llama3.2"           # or mistral, phi3, qwen2.5


CFG = EvalConfig()

# Valid role labels — must match your RoleLabelEncoder exactly
VALID_ROLES = ["adversary", "ally", "hero", "neutral", "villain"]


# =============================================================================
# PROMPT BUILDER
# =============================================================================

SYSTEM_PROMPT = """You are a literary analyst specializing in narrative role classification.
Your task is to classify a character's narrative role based on excerpts from their appearances in a novel.

The five possible roles are:
  - hero: The protagonist or main heroic figure driving the story forward
  - villain: The primary antagonist actively working against the hero or causing harm
  - ally: A supporting character who aids or supports the hero
  - adversary: A character who opposes or creates obstacles, but is not the main villain
  - neutral: A character who does not clearly align with hero or villain forces

Rules:
  - Respond with ONLY one of these exact words: hero, villain, ally, adversary, neutral
  - Do not explain your reasoning
  - Do not add punctuation or capitalization
  - If uncertain, choose the closest match — never respond with anything outside the five labels"""


def _build_full_context(character_name: str, chunks: dict) -> str:
    """
    Full condition: no truncation. Concatenate all three raw chunks.
    Claude/GPT/Gemini have 128k-200k+ context so this fits trivially.
    Ollama models vary — see OllamaEvaluator note.
    """
    chunk_keys = ["first", "last", "frequent"]
    labels = {
        "first":    "First appearance",
        "last":     "Last appearance",
        "frequent": "Key scene",
    }
    parts = [f"Character: {character_name}"]
    for k in chunk_keys:
        text = chunks.get(k, {}).get("text", "")
        if text:
            parts.append(f"{labels[k]}: {text}")
    return " [SEP] ".join(parts)


def build_prompt(
    character_name: str,
    chunks: dict,
    tokenizer,
    max_length: Optional[int],
) -> str:
    """
    Builds the user-facing prompt.

    fair condition (max_length=4096):
        Uses budget_chunks() from dataset.py — identical token budget and
        truncation logic as BERT/Longformer. The only variable across models
        is the model itself, not the input text.

    full condition (max_length=None):
        Skips budgeting entirely. LLMs receive the complete raw chunks.
        Shows what cloud LLMs achieve with their full context advantage.
        Also reveals whether small local models (Ollama) can exploit it.
    """
    if max_length is None:
        context = _build_full_context(character_name, chunks)
    else:
        context = budget_chunks(chunks, character_name, tokenizer, max_length)

    return f"Classify the narrative role of this character:\n\n{context}"


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def parse_role(response_text: str) -> Optional[str]:
    """
    Robustly extract a valid role from LLM output.
    Handles capitalization, punctuation, and brief explanations that
    appear despite instructions (e.g. "The role is: hero." → "hero").
    """
    if not response_text:
        return None

    text = response_text.strip().lower()

    # Direct match first
    if text in VALID_ROLES:
        return text

    # Scan for first valid role word anywhere in the response
    for role in VALID_ROLES:
        if role in text:
            return role

    return None  # genuinely unparseable


# =============================================================================
# BASE EVALUATOR
# =============================================================================

class BaseLLMEvaluator:
    """
    Base class for all LLM evaluators.
    Subclasses implement _call_api() — rate limiting, retry, and prompt
    building are all handled here so each subclass is minimal.
    """

    def __init__(self, model_name: str, rpm_limit: int, cfg: EvalConfig):
        self.model_name     = model_name
        self.min_interval   = 60.0 / rpm_limit
        self.cfg            = cfg
        self._last_call     = 0.0
        self.total_cost_usd = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        wait    = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _call_api(self, system: str, user: str) -> str:
        raise NotImplementedError

    def classify(
        self,
        character_name: str,
        chunks: dict,
        tokenizer,
        max_length: Optional[int],
    ) -> Optional[str]:
        """
        Full pipeline: rate limit → build prompt → call API → parse → retry.
        max_length flows from cfg.active_max_length — None for full, 4096 for fair.
        """
        user_prompt = build_prompt(character_name, chunks, tokenizer, max_length)

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                self._rate_limit()
                raw  = self._call_api(SYSTEM_PROMPT, user_prompt)
                role = parse_role(raw)

                if role:
                    return role
                else:
                    print(f"    ⚠ Unparseable (attempt {attempt}): '{raw[:80]}'")

            except Exception as e:
                print(f"    ✗ API error (attempt {attempt}/{self.cfg.max_retries}): {e}")
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_delay * attempt)  # exponential backoff

        return None  # all retries exhausted


# =============================================================================
# CLAUDE EVALUATOR
# =============================================================================

class ClaudeEvaluator(BaseLLMEvaluator):
    """
    Anthropic Claude API.
    Context window: 200k tokens — full condition fits comfortably.
    Set ANTHROPIC_API_KEY in environment or Colab Secrets.
    """

    _INPUT_COST_PER_1K  = 0.003    # approximate for claude-sonnet-4-6
    _OUTPUT_COST_PER_1K = 0.015

    def __init__(self, cfg: EvalConfig):
        super().__init__(cfg.claude_model, cfg.claude_rpm, cfg)
        import anthropic
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        print(f"✓ Claude ready  ({cfg.claude_model})")

    def _call_api(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=10,      # one-word answer — minimizes cost
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        usage = response.usage
        self.total_cost_usd += (
            usage.input_tokens  / 1000 * self._INPUT_COST_PER_1K +
            usage.output_tokens / 1000 * self._OUTPUT_COST_PER_1K
        )
        return response.content[0].text


# =============================================================================
# GPT EVALUATOR
# =============================================================================

class GPTEvaluator(BaseLLMEvaluator):
    """
    OpenAI Chat Completions API.
    Context window: 128k tokens — full condition fits comfortably.
    Set OPENAI_API_KEY in environment or Colab Secrets.
    """

    _INPUT_COST_PER_1K  = 0.00015  # approximate for gpt-4o-mini
    _OUTPUT_COST_PER_1K = 0.0006

    def __init__(self, cfg: EvalConfig):
        super().__init__(cfg.gpt_model, cfg.gpt_rpm, cfg)
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print(f"✓ GPT ready     ({cfg.gpt_model})")

    def _call_api(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=10,
            temperature=0,      # deterministic — classification task
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        usage = response.usage
        self.total_cost_usd += (
            usage.prompt_tokens     / 1000 * self._INPUT_COST_PER_1K +
            usage.completion_tokens / 1000 * self._OUTPUT_COST_PER_1K
        )
        return response.choices[0].message.content


# =============================================================================
# GEMINI EVALUATOR
# =============================================================================

class GeminiEvaluator(BaseLLMEvaluator):

    def __init__(self, cfg: EvalConfig):
        super().__init__(cfg.gemini_model, cfg.gemini_rpm, cfg)
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.types = types
        print(f"✓ Gemini ready  ({cfg.gemini_model})")

    def _call_api(self, system: str, user: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user,
            config=self.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=10,
                temperature=0,
            ),
        )
        return response.text

# =============================================================================
# OLLAMA EVALUATOR — LOCAL ONLY, NOT COLAB
# =============================================================================
# SETUP (on your local machine):
#   1. Install: https://ollama.com/download
#   2. Pull a model (choose based on your VRAM):
#        ollama pull llama3.2    ← 2B,  ~2GB VRAM, fast
#        ollama pull phi3        ← 3.8B, ~4GB VRAM, good balance
#        ollama pull mistral     ← 7B,  ~8GB VRAM, stronger
#        ollama pull qwen2.5     ← 7B,  ~8GB VRAM, strong multilingual
#   3. Server starts automatically on http://localhost:11434
#   4. Run:
#        python eval_llm.py --models ollama --ollama-model llama3.2 --condition fair
#        python eval_llm.py --models ollama --ollama-model llama3.2 --condition full
#
# Note on full condition for Ollama:
#   Small models (2B-7B) often degrade on very long inputs even within their
#   context window. The fair vs full delta for Ollama is worth reporting in
#   your paper — it directly shows whether small local models can exploit
#   long context the way large cloud models can.

class OllamaEvaluator(BaseLLMEvaluator):

    def __init__(self, cfg: EvalConfig):
        super().__init__(cfg.ollama_model, cfg.ollama_rpm, cfg)
        import ollama
        self.client = ollama
        try:
            self.client.list()
            print(f"✓ Ollama ready  ({cfg.ollama_model})")
        except Exception:
            raise RuntimeError(
                "\nOllama server not running.\n"
                "Start it:   ollama serve\n"
                "Pull model: ollama pull llama3.2"
            )

    def _call_api(self, system: str, user: str) -> str:
        response = self.client.chat(
            model=self.model_name,
            options={"temperature": 0, "num_predict": 10},
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response["message"]["content"]


# =============================================================================
# EVALUATION LOOP
# =============================================================================

def evaluate_model(
    evaluator: BaseLLMEvaluator,
    data: list,
    label_encoder: RoleLabelEncoder,
    tokenizer,
    cfg: EvalConfig,
) -> dict:
    """
    Mirrors the K-fold structure of BERT/Longformer but skips training.

    For each fold:
      - Training split is ignored (LLMs need no fine-tuning)
      - Val split receives zero-shot prompts and is classified
      - Per-fold F1 scores are computed identically to fine-tuned runs

    Because StratifiedKFold uses the same n_folds=5 and seed=42 as your
    BERT/Longformer scripts, the val splits are identical — aggregate
    mean ± std is directly comparable across all model types.
    """
    all_labels  = label_encoder.transform([d["role"] for d in data])
    data_array  = np.array(data, dtype=object)
    label_names = label_encoder.label_names
    max_length  = cfg.active_max_length   # None for full, 4096 for fair

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    fold_metrics       = []
    all_error_analyses = []
    parse_failures     = 0
    total_evaluated    = 0

    for fold, (_, val_idx) in enumerate(
        skf.split(data_array, all_labels), start=1
    ):
        print(f"\n{'='*60}")
        print(f"  {evaluator.model_name} [{cfg.condition}] — FOLD {fold}/{cfg.n_folds}")
        print(f"{'='*60}")

        val_data = [data[i] for i in val_idx]
        context_label = "UNCAPPED" if max_length is None else f"{max_length} tokens"
        print(f"  Val: {len(val_data)} characters | Context: {context_label}")

        preds_fold  = []
        labels_fold = []

        for entry in val_data:
            true_role = entry["role"]
            true_id   = label_encoder.transform([true_role])[0]

            pred_role = evaluator.classify(
                entry["character"],
                entry.get("chunks", {}),
                tokenizer,
                max_length,         # None for full condition, 4096 for fair
            )

            if pred_role is None:
                print(f"    ✗ Parse failure: {entry['character']} ({entry['novel']})")
                parse_failures += 1
                pred_role = "neutral"   # fallback — keeps metrics from crashing

            pred_id = label_encoder.transform([pred_role])[0]
            preds_fold.append(pred_id)
            labels_fold.append(true_id)
            total_evaluated += 1

        # --- Fold metrics ---
        f1_macro    = f1_score(labels_fold, preds_fold, average="macro",    zero_division=0)
        f1_micro    = f1_score(labels_fold, preds_fold, average="micro",    zero_division=0)
        f1_weighted = f1_score(labels_fold, preds_fold, average="weighted", zero_division=0)

        fold_metrics.append({
            "eval_f1_macro":    f1_macro,
            "eval_f1_micro":    f1_micro,
            "eval_f1_weighted": f1_weighted,
        })

        print(f"  F1 Macro:    {f1_macro:.4f}")
        print(f"  F1 Micro:    {f1_micro:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")

        # --- Error analysis (same format as BERT/Longformer) ---
        report = classification_report(
            labels_fold, preds_fold, target_names=label_names, zero_division=0
        )
        print(f"\n--- Fold {fold} Classification Report ---")
        print(report)

        cm    = confusion_matrix(labels_fold, preds_fold)
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        print(f"Confusion Matrix:\n{cm_df}\n")

        misclassified = []
        for i, (pred, label) in enumerate(zip(preds_fold, labels_fold)):
            if pred != label:
                e = val_data[i]
                misclassified.append({
                    "character": e["character"],
                    "novel":     e["novel"],
                    "true_role": label_names[label],
                    "predicted": label_names[pred],
                })

        all_error_analyses.append({
            "report":           report,
            "confusion_matrix": cm.tolist(),
            "misclassified":    misclassified,
        })

    # --- Aggregate ---
    print(f"\n{'='*60}")
    print(f"  AGGREGATE — {evaluator.model_name} [{cfg.condition}] ({cfg.n_folds}-Fold CV)")
    print(f"{'='*60}")

    summary = {}
    for key in ["eval_f1_macro", "eval_f1_micro", "eval_f1_weighted"]:
        values = [m[key] for m in fold_metrics]
        mean, std = np.mean(values), np.std(values)
        summary[key] = {"mean": round(mean, 4), "std": round(std, 4), "per_fold": values}
        print(f"  {key}: {mean:.4f} ± {std:.4f}")

    failure_rate = parse_failures / max(total_evaluated, 1)
    print(f"\n  Parse failures: {parse_failures}/{total_evaluated} ({failure_rate:.1%})")
    if hasattr(evaluator, "total_cost_usd"):
        print(f"  Estimated cost: ${evaluator.total_cost_usd:.4f} USD")

    return {
        "model":              evaluator.model_name,
        "approach":           "zero-shot",
        "condition":          cfg.condition,        # "fair" or "full" — self-documenting
        "max_length":         max_length,           # None if full, 4096 if fair
        "n_folds":            cfg.n_folds,
        "parse_failures":     parse_failures,
        "total_evaluated":    total_evaluated,
        "estimated_cost_usd": round(getattr(evaluator, "total_cost_usd", 0.0), 4),
        "summary":            summary,
        "per_fold":           fold_metrics,
        "error_analyses":     all_error_analyses,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

EVALUATOR_MAP = {
    "claude": lambda cfg: ClaudeEvaluator(cfg),
    "gpt":    lambda cfg: GPTEvaluator(cfg),
    "gemini": lambda cfg: GeminiEvaluator(cfg),
    "ollama": lambda cfg: OllamaEvaluator(cfg),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=["claude", "gpt", "gemini", "ollama"],
        default=["claude"],
    )
    parser.add_argument(
        "--condition", choices=["fair", "full"], default="fair",
        help=(
            "fair: same 4096-token budget as Longformer (controls for input). "
            "full: uncapped — LLMs receive complete raw chunks."
        ),
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="1 fold only — verify pipeline before committing to full run",
    )
    parser.add_argument(
        "--ollama-model", type=str, default=CFG.ollama_model,
        help="Ollama model name (e.g. llama3.2, mistral, phi3)",
    )
    args = parser.parse_args()

    # Apply args to config
    CFG.condition    = args.condition
    CFG.output_dir   = f"outputs/llm_eval_{args.condition}"
    CFG.ollama_model = args.ollama_model

    if args.smoke_test:
        CFG.n_folds = 1
        print("🔥 SMOKE TEST — 1 fold only\n")

    os.makedirs(CFG.output_dir, exist_ok=True)

    print(f"Condition : {CFG.condition.upper()}")
    print(f"Max length: {CFG.active_max_length or 'UNCAPPED (full context)'}")
    print(f"Output dir: {CFG.output_dir}\n")

    # --- Load data ---
    data = load_chunked_data(CFG.chunks_json)
    label_encoder = RoleLabelEncoder()
    label_encoder.fit([d["role"] for d in data])
    label_encoder.save(f"{CFG.output_dir}/label_encoder.json")

    print(f"Loaded {len(data)} characters")
    print(f"Labels: {label_encoder.label_names}")
    dist = Counter(d["role"] for d in data)
    for role, count in sorted(dist.items()):
        print(f"  {role}: {count}")

    # Tokenizer is used ONLY by budget_chunks() for token counting in the
    # fair condition. LLMs receive raw text strings — never token IDs.
    # In the full condition, build_prompt() bypasses budget_chunks() entirely,
    # but we still load the tokenizer since it's shared infrastructure.
    print("\nLoading tokenizer for chunk budgeting...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

    # --- Run each model ---
    all_results = {}

    for model_key in args.models:
        print(f"\n{'#'*60}")
        print(f"  Evaluating: {model_key.upper()}  [{CFG.condition}]")
        print(f"{'#'*60}")

        try:
            evaluator = EVALUATOR_MAP[model_key](CFG)
        except Exception as e:
            print(f"  ✗ Failed to initialize {model_key}: {e}")
            continue

        results = evaluate_model(evaluator, data, label_encoder, tokenizer, CFG)
        all_results[model_key] = results

        # Save per-model immediately — a later API failure won't lose this data
        model_path = f"{CFG.output_dir}/{model_key}_results.json"
        with open(model_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved → {model_path}")

    # Save combined file for easy Step 3 ingestion
    combined_path = f"{CFG.output_dir}/all_llm_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Combined results → {combined_path}")


if __name__ == "__main__":
    main()