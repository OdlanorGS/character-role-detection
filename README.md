# Character Role Detection

Classifies fictional characters in literary texts as **hero, villain, ally, adversary, or neutral** using transformer-based models. Character mentions are extracted via coreference resolution chains, chunked into context windows, and fed into fine-tuned BERT or Longformer classifiers. Zero-shot LLM baselines (Claude, GPT-4, Gemini) are included for comparison.

## Repository Structure

```
character-role-detection/
├── baseclean.py                  # Generate annotation templates from coreference chains
├── merging_annotations.py        # Merge individual annotations into master CSV
├── annotations_templates/        # Per-novel role annotation CSVs (Novel1–Novel30)
├── code/
│   ├── dataset.py                # Dataset class, label encoder, data loading
│   ├── model_bert.py             # BERT classifier with weighted loss
│   ├── model_longformer.py       # Longformer classifier with global attention
│   ├── train_bert.py             # BERT training with 5-fold cross-validation
│   ├── train_longformer.py       # Longformer training (4096 tokens)
│   ├── train_longformer_1024.py  # Longformer variant (1024 tokens)
│   ├── train_longformer_2048.py  # Longformer variant (2048 tokens)
│   ├── step_2_1_chunking.py      # Extract first/last/frequent mention chunks
│   └── eval_llm.py               # Zero-shot LLM evaluation
├── outputs/
│   ├── chunked_characters.json   # Processed characters with context chunks
│   ├── bert/                     # BERT checkpoints and results
│   └── longformer/               # Longformer checkpoints and results
└── results/                      # Additional result dumps
```

## Data Pipeline

1. **Coreference resolution** — extract character mention chains from novels
2. **`baseclean.py`** — generate per-novel annotation template CSVs
3. **`merging_annotations.py`** — consolidate annotations into `master_roles.csv`
4. **`code/step_2_1_chunking.py`** — extract three chunks per character (first, last, most frequent mention) each with a ~2000-character context window
5. **`code/train_bert.py` / `train_longformer.py`** — train classifiers with 5-fold stratified cross-validation

## Installation

```bash
pip install torch transformers scikit-learn pandas numpy tqdm anthropic openai google-generativeai ollama python-dotenv
```

## Usage

```bash
# Step 1: generate annotation templates
python baseclean.py

# Step 2: merge completed annotations
python merging_annotations.py

# Step 3: chunk character mentions
python code/step_2_1_chunking.py

# Step 4a: train BERT classifier
python code/train_bert.py

# Step 4b: train Longformer classifier
python code/train_longformer.py

# Step 5: zero-shot LLM evaluation (requires .env with API keys)
python code/eval_llm.py
```

## Environment Variables

Create a `.env` file in the project root for LLM evaluation:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

## Results

| Model | Token Limit | Weighted F1 |
|---|---|---|
| BERT | 512 | ~0.33 |
| Longformer | 1024 | — |
| Longformer | 2048 | — |
| Longformer | 4096 | — |
| Claude (zero-shot) | 4096 (fair) / full | — |
| GPT-4 (zero-shot) | 4096 (fair) / full | — |
| Gemini (zero-shot) | 4096 (fair) / full | — |

Classes: `hero`, `villain`, `ally`, `adversary`, `neutral`. Class imbalance is handled via inverse-frequency weighting in the loss function.
