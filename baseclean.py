import re
import csv
from pathlib import Path

def parse_coref_line(line: str):
    """
    Expected patterns like:
    '1 1 | Barold | He | him | ...'
    Returns (animacy, character, mentions_list)
    """
    parts = [p.strip() for p in line.strip().split("|")]
    if not parts or not parts[0]:
        return None

    # First chunk contains the 2 integers, e.g. "1 1"
    nums = re.findall(r"\d+", parts[0])
    if len(nums) < 2:
        return None
    animacy = int(nums[0])
    character = int(nums[1])

    # Remaining chunks are mentions
    mentions = [m.strip() for m in parts[1:] if m.strip()]
    return animacy, character, mentions

def pick_canonical_name(mentions):
    """
    Heuristic: prefer longer, name-like mentions (contains capital letter, not just pronoun).
    """
    if not mentions:
        return ""
    pronouns = {"he","him","his","she","her","hers","they","them","their","theirs","i","me","my","mine","you","your","yours","it","its"}
    candidates = []
    for m in mentions:
        m_clean = m.strip().strip(",.?!;:\"'").lower()
        if m_clean in pronouns:
            continue
        # prefer mentions that look like names/titles
        score = 0
        if any(ch.isupper() for ch in m): score += 2
        if "mr" in m_clean or "mrs" in m_clean or "miss" in m_clean or "lord" in m_clean or "lady" in m_clean: score += 2
        score += min(len(m), 30) / 30  # slight preference for longer
        candidates.append((score, m))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return mentions[0]

def build_template(coref_path: Path, out_csv: Path, doc_id: str):
    rows = []
    with coref_path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parsed = parse_coref_line(line)
            if not parsed:
                continue
            animacy, character, mentions = parsed
            canonical = pick_canonical_name(mentions)
            rows.append({
                "doc_id": doc_id,
                "chain_id": i,
                "animacy": animacy,
                "character": character,
                "canonical_name": canonical,
                "mentions": " | ".join(mentions),
                "role": "",          # YOU FILL THIS IN
                "confidence": "",    # YOU FILL THIS IN
                "notes": ""          # YOU FILL THIS IN
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    # EDIT THESE PATHS
    coref_dir = Path(r"D:/University FIU/CAPSTONE/character_v1.0.0/Data/AnnotatedCENData/CENCoref") # folder with your 30 coref files
    out_dir = Path("annotations_templates")      # output folder

    for coref_file in sorted(coref_dir.glob("*.txt")):
        doc_id = coref_file.stem  # e.g., "Novel1" or whatever your filenames are
        out_csv = out_dir / f"{doc_id}_roles.csv"
        build_template(coref_file, out_csv, doc_id)
        print("Wrote:", out_csv)
