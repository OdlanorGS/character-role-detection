"""
Step 2.1: First / Last / Frequent Mention Chunking
Uses coreference chains to locate characters in source text,
then extracts context windows for downstream classification.

Usage:
    python step_2_1_chunking.py

Outputs:
    - outputs/chunked_characters.json   (structured chunks per character)
    - outputs/chunk_statistics.csv       (coverage & mention stats)
    - Console diagnostics for debugging mention matching
"""

import os
import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ChunkConfig:
    # --- Paths ---
    data_csv: str = "D:\\University FIU\\CAPSTONE\\annotations_templates\\master_roles.csv"
    texts_dir: str = "D:\\University FIU\\CAPSTONE\\character_v1.0.0\\Data\\AnnotatedCENData\\CENTexts"
    output_dir: str = "outputs/"

    # --- CSV parsing ---
    csv_separator: str = ","
    mention_separator: str = "|"

    # --- Roles to keep ---
    valid_roles: tuple = ("hero", "villain", "ally", "adversary", "neutral")

    # --- Chunk extraction ---
    window_chars: int = 2000 # target chunk size (characters)
    min_mention_length: int = 3 # minimum length for a mention to be considered (filters out very short ones)
    pronoun_filter: bool = True # whether to exclude pronouns from mention matching
    mark_anchor: bool = True                # wrap anchor mention with [CHAR]...[/CHAR]
    pronouns: tuple = (
        "i", "me", "my", "mine", "myself",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "they", "them", "their", "theirs", "themselves",
        "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself",
        "it", "its", "itself",
    ) # common English pronouns to filter out from mention matching


CFG = ChunkConfig()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_annotated_data(cfg: ChunkConfig) -> pd.DataFrame:
    """Load and filter the annotated character CSV."""
    df = pd.read_csv(
        cfg.data_csv,
        sep=",",
        encoding="cp1252"
    )

    df["role"] = df["role"].astype(str).str.strip().str.lower() # ensure role column is clean and lowercase
    df = df[df["role"].isin(cfg.valid_roles)].copy() # keep only rows with valid roles

    df["mention_list"] = df["mentions"].fillna("").apply(
        lambda x: [m.strip() for m in x.split(cfg.mention_separator) if m.strip()]
    ) # convert mentions string to list, stripping whitespace and filtering out empty mentions

    print(f"Loaded {len(df)} characters across {df['novel'].nunique()} novels") # for diagnostic purposes
    print(f"Role distribution:\n{df['role'].value_counts().to_string()}\n") # for diagnostic purposes
    return df


def load_novel_text(novel_id: str, cfg: ChunkConfig) -> Optional[str]:

    """Load the raw text file for a novel with encoding fallback."""
    
    path = Path(cfg.texts_dir) / f"{novel_id}.txt"
    if not path.exists(): # diagnostic warning if text file is missing
        print(f"  WARNING: Text not found for {novel_id} at {path}")
        return None

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"] # common encodings to try for legacy text files
    last_error = None # keep track of last decode error for diagnostics

    for enc in encodings_to_try: # try each encoding until one works, had some issues
        try:
            text = path.read_text(encoding=enc)
            print(f"  Loaded {novel_id}.txt using {enc}")
            return text
        except UnicodeDecodeError as e:
            last_error = e
            continue

    print(f"  WARNING: Could not decode text for {novel_id} at {path}") # if all encodings fail, log a warning and return None
    if last_error:
        print(f"    Last decode error: {last_error}")
    return None


# =============================================================================
# MENTION MATCHING
# =============================================================================

def filter_mentions(mentions: List[str], cfg: ChunkConfig) -> List[str]:
    """
    Filter out pronouns and very short mentions.
    Only named/descriptive mentions are used for locating positions.
    """
    filtered = []
    for m in mentions:
        m_clean = m.strip()
        if len(m_clean) < cfg.min_mention_length:
            continue
        if cfg.pronoun_filter and m_clean.lower() in cfg.pronouns:
            continue
        filtered.append(m_clean)
    return filtered


def find_mention_positions(text: str, mentions: List[str]) -> List[Dict]:
    """
    Locate all occurrences of each mention in the text.

    Uses word-boundary matching to avoid partial matches. For example, "Barold" should not match "Barold's" or "Baroldia".

    Returns sorted list of {mention, position, end} dicts.

    """
    positions = []
    seen = set()

    for mention in mentions: # use regex word boundaries to find exact matches of the mention in the text
        pattern = r'\b' + re.escape(mention) + r'\b'
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                if pos not in seen:
                    seen.add(pos)
                    positions.append({
                        "mention": mention, # original mention text (not lowercased, to preserve formatting for marking)
                        "position": pos, # character index of the mention in the text
                        "end": match.end(), # character index of the end of the mention in the text
                    })
        except re.error:
            continue

    positions.sort(key=lambda x: x["position"])
    return positions


# =============================================================================
# CHUNK EXTRACTION (FIXED — ANCHOR-PRESERVING)
# =============================================================================

def find_sentence_bounds(text: str, anchor_pos: int) -> Tuple[int, int]:
    """
    Find the sentence boundaries that contain anchor_pos.
    Uses ., !, ?, or paragraph breaks as rough sentence delimiters.
    Returns (sentence_start, sentence_end).
    
    This guarantees the anchor mention's sentence is always identified
    so it can be preserved in the final chunk.
    """
    # --- find sentence start ---
    sentence_start = 0
    i = anchor_pos

    while i > 0:
        # paragraph break
        if i >= 2 and text[i-2:i] == "\n\n":
            sentence_start = i
            break
        # punctuation boundary
        if text[i-1] in ".!?":
            sentence_start = i
            while sentence_start < len(text) and text[sentence_start] in " \n\r\t":
                sentence_start += 1
            break
        i -= 1

    # --- find sentence end ---
    sentence_end = len(text)
    i = anchor_pos

    while i < len(text):
        # paragraph break
        if i + 2 <= len(text) and text[i:i+2] == "\n\n":
            sentence_end = i
            break
        # punctuation boundary
        if text[i] in ".!?":
            sentence_end = i + 1
            break
        i += 1

    return sentence_start, sentence_end


def expand_to_boundaries(text: str, start: int, end: int) -> Tuple[int, int]:
    """
    Nudge chunk edges outward so they don't cut mid-word.
    """
    while start > 0 and text[start - 1] not in " \n\r\t": # expand left until hitting whitespace or start of text
        start -= 1
    while end < len(text) and text[end:end+1] not in ["", " ", "\n", "\r", "\t"]: # expand right until hitting whitespace or end of text
        end += 1
    return start, end


def extract_chunk(text: str, center_pos: int, window: int) -> Dict:
    """
    Extract a chunk that ALWAYS preserves the sentence containing the anchor.
    Then expands symmetrically around that sentence up to `window` characters.

    Old behavior (buggy) from initial_chunking_techniqque.py:
        1. Center a raw window on the anchor position
        2. Nudge start forward to a sentence boundary
         Could trim past the anchor sentence entirely

    New behavior (fixed):
        1. Find the sentence containing the anchor
        2. Keep that sentence no matter what
        3. Expand left and right equally until hitting the window size
        Anchor sentence is always inside the chunk
    """
    sent_start, sent_end = find_sentence_bounds(text, center_pos)
    anchor_sentence_len = sent_end - sent_start

    # If anchor sentence alone exceeds window, just use it
    if anchor_sentence_len >= window:
        start, end = sent_start, sent_end
    else:
        remaining = window - anchor_sentence_len
        left_extra = remaining // 2
        right_extra = remaining - left_extra

        start = max(0, sent_start - left_extra)
        end = min(len(text), sent_end + right_extra)

        # Don't cut mid-word at edges
        start, end = expand_to_boundaries(text, start, end)

    chunk_text = text[start:end].strip()

    return {
        "text": chunk_text,
        "start": start,
        "end": end,
        "length": len(chunk_text),
    }


# =============================================================================
# ANCHOR VALIDATION & MARKING
# =============================================================================

def anchor_present(chunk_text: str, anchor_mention: str) -> bool:
    """Check if the anchor mention still exists inside the extracted chunk."""
    return anchor_mention.lower() in chunk_text.lower()


def mark_anchor_in_text(chunk_text: str, anchor_mention: str) -> str:
    """
    Wrap the first occurrence of the anchor mention with [CHAR]...[/CHAR].

    This helps the model know which character in the scene is the target, especially in chunks where multiple characters interact.

    Example:
        "they found Barold's eyes fixed upon her"
        → "they found [CHAR] Barold [/CHAR]'s eyes fixed upon her"
    """
    pattern = re.escape(anchor_mention) # match the anchor mention exactly (case-insensitive)
    return re.sub(
        pattern,
        f"[CHAR] {anchor_mention} [/CHAR]", # add markers around the anchor mention
        chunk_text,
        count=1,
        flags=re.IGNORECASE, 
    )


# =============================================================================
# CORE CHUNKING LOGIC
# =============================================================================

def get_character_chunks(
    text: str,
    mentions: List[str],
    cfg: ChunkConfig,
) -> Dict:
    """
    Core chunking logic for one character.
    Returns first, last, and most-frequent mention chunks
    with anchor validation and optional [CHAR] marking.
    """
    named_mentions = filter_mentions(mentions, cfg) # filter out pronouns and short mentions to focus on named/descriptive ones for position finding

    if not named_mentions: # if no valid named mentions remain after filtering, we can't reliably locate the character in the text, so we skip chunking for this character
        return {
            "status": "no_named_mentions", 
            "chunks": {}, 
            "stats": {"total_mentions": len(mentions), "named_mentions": 0},
        }

    positions = find_mention_positions(text, named_mentions)

    if not positions: # if we couldn't find any positions for the mentions in the text, we can't extract chunks, so we return a status indicating this along with stats about the mentions
        return {
            "status": "no_positions_found",
            "chunks": {},
            "stats": {
                "total_mentions": len(mentions), # total number of mentions provided for this character (including pronouns and short ones)
                "named_mentions": len(named_mentions), # number of mentions that passed the filtering (used for position finding)
                "named_list": named_mentions[:10], # sample of the named mentions for diagnostics (up to 10)
            },
        }

    # --- FIRST mention ---
    first = positions[0]

    # --- LAST mention ---
    last = positions[-1]

    # --- MOST FREQUENT (median occurrence) ---
    mention_counts = Counter(p["mention"] for p in positions) # count how many times each mention appears in the text
    most_common_mention = mention_counts.most_common(1)[0] # get the most common mention and its count (e.g. ("Barold", 15))
    freq_positions = [p for p in positions if p["mention"] == most_common_mention[0]] # get all positions of the most common mention
    mid_idx = len(freq_positions) // 2 #
    frequent = freq_positions[mid_idx] #

    # --- Build chunks with validation ---
    raw_chunks = {
        "first":    (first, {}), # for the first mention, we don't have extra metadata to add to the stats
        "last":     (last, {}),
        "frequent": (frequent, {"mention_count": most_common_mention[1]}),
    }

    chunks = {}
    anchor_warnings = []

    for chunk_type, (anchor_info, extra_meta) in raw_chunks.items():
        chunk = extract_chunk(text, anchor_info["position"], cfg.window_chars)

        # Validate anchor is still in the chunk
        found = anchor_present(chunk["text"], anchor_info["mention"])
        if not found:
            anchor_warnings.append(
                f"anchor '{anchor_info['mention']}' not found in "
                f"{chunk_type} chunk at position {anchor_info['position']}"
            )

        # Mark the anchor character in the text
        if cfg.mark_anchor and found:
            chunk["text"] = mark_anchor_in_text(
                chunk["text"], anchor_info["mention"]
            )

        chunks[chunk_type] = {
            **chunk,
            "anchor_mention": anchor_info["mention"],
            "anchor_position": anchor_info["position"],
            "anchor_found_in_text": found,
            **extra_meta,
        }

    # Log any anchor warnings
    for warning in anchor_warnings:
        print(f"    WARNING: {warning}")

    # --- Stats ---
    positions_set = {c["anchor_position"] for c in chunks.values()}

    stats = {
        "total_mentions": len(mentions), #
        "named_mentions": len(named_mentions), # number of mentions that were considered for position finding (after filtering)
        "positions_found": len(positions),
        "unique_chunk_anchors": len(positions_set),
        "text_length": len(text),
        "coverage_pct": round(
            (sum(c["length"] for c in chunks.values()) / len(text)) * 100, 2
        ),
        "most_common_mention": most_common_mention[0],
        "most_common_count": most_common_mention[1],
        "first_position_pct": round((first["position"] / len(text)) * 100, 2),
        "last_position_pct": round((last["position"] / len(text)) * 100, 2),
        "all_anchors_found": len(anchor_warnings) == 0,
        "anchor_warnings": anchor_warnings,
    }

    return {"status": "ok", "chunks": chunks, "stats": stats}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_chunking(cfg: ChunkConfig):
    """Run the full chunking pipeline and save results."""
    os.makedirs(cfg.output_dir, exist_ok=True) 

    df = load_annotated_data(cfg) # load the annotated character data from the CSV, filtering to only include rows with valid roles and converting the mentions string into a list of mentions for each character

    results = []
    stats_rows = []
    total_anchor_warnings = 0

    for novel_id in sorted(df["novel"].unique()):
        text = load_novel_text(novel_id, cfg)
        if text is None:
            continue

        novel_chars = df[df["novel"] == novel_id]
        print(f"\n--- {novel_id} ({len(novel_chars)} characters, "
              f"{len(text):,} chars) ---") # diagnostic header for each novel showing how 
                                           # many characters are being processed and the length of the text

        for _, row in novel_chars.iterrows():
            result = get_character_chunks(text, row["mention_list"], cfg)

            char_entry = {
                "novel": novel_id,
                "character": row["canonical_name"],
                "role": row["role"],
                "chain_id": int(row["chain_id"]),
                "status": result["status"],
                "chunks": {},
                "stats": result["stats"],
            }

            if result["status"] == "ok":
                for chunk_type in ["first", "last", "frequent"]:
                    chunk = result["chunks"][chunk_type]
                    char_entry["chunks"][chunk_type] = {
                        "text": chunk["text"],
                        "anchor_mention": chunk["anchor_mention"],
                        "anchor_position": chunk["anchor_position"],
                        "anchor_found_in_text": chunk["anchor_found_in_text"],
                        "length": chunk["length"],
                    }

                anchors_ok = result["stats"]["all_anchors_found"]
                n_warnings = len(result["stats"]["anchor_warnings"])
                total_anchor_warnings += n_warnings

                status_icon = "OK" if anchors_ok else "OK*"
                extra = (f"coverage={result['stats']['coverage_pct']}%, "
                         f"positions={result['stats']['positions_found']}")
                if not anchors_ok:
                    extra += f", anchor_warnings={n_warnings}"
            else:
                status_icon = "SKIP"
                extra = result["status"]

            print(f"  [{status_icon}] {row['canonical_name']:<40} "
                  f"role={row['role']:<12} {extra}")

            results.append(char_entry)

            # Stats row (exclude non-serializable fields)
            stat_row = {
                "novel": novel_id,
                "character": row["canonical_name"],
                "role": row["role"],
            }
            for k, v in result["stats"].items():
                if k != "anchor_warnings":
                    stat_row[k] = v
            stats_rows.append(stat_row)

    # --- Save outputs ---
    json_path = Path(cfg.output_dir) / "chunked_characters.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} character chunks → {json_path}")

    stats_df = pd.DataFrame(stats_rows)
    stats_path = Path(cfg.output_dir) / "chunk_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved chunk statistics → {stats_path}")

    # --- Summary ---
    ok_count = sum(1 for r in results if r["status"] == "ok")
    skip_count = len(results) - ok_count
    print(f"\n{'='*60}")
    print(f"  CHUNKING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total characters:     {len(results)}")
    print(f"  Successfully chunked: {ok_count}")
    print(f"  Skipped (no match):   {skip_count}")
    print(f"  Anchor warnings:      {total_anchor_warnings}")

    if not stats_df.empty and "coverage_pct" in stats_df.columns:
        valid = stats_df[stats_df["coverage_pct"].notna()]
        if not valid.empty:
            print(f"  Avg coverage/char:    {valid['coverage_pct'].mean():.1f}%")
            print(f"  Avg positions found:  {valid['positions_found'].mean():.1f}")

    if total_anchor_warnings > 0:
        print(f"\n  ⚠ {total_anchor_warnings} chunks lost their anchor mention.")
        print(f"    Check chunk_statistics.csv for all_anchors_found=False rows.")
    else:
        print(f"\n  ✓ All anchor mentions preserved in their chunks.")

    return results, stats_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results, stats = run_chunking(CFG)