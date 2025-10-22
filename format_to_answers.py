import pandas as pd
import numpy as np
from collections import defaultdict

# --- paths ---
RESP_PATH = "data/Antworten.csv"
META_PATH = "data/Variablen_Befragung_55_plus_2023.csv"
OUT_PATH  = "data/format_to_answers.csv"

# --- read ---
df_resp = pd.read_csv(RESP_PATH)
df_meta = pd.read_csv(META_PATH)

# --- normalize column names just in case ---
df_meta["FrageName"] = df_meta["FrageName"].astype(str).str.strip()
df_meta["Format"]    = df_meta["Format"].astype(str).str.strip()

# We only keep questions that exist in the responses table
questions = [q for q in df_meta["FrageName"].unique() if q in df_resp.columns]
df_meta = df_meta[df_meta["FrageName"].isin(questions)].copy()

# --- cleaning helper for "no info" tokens ---
NOINFO = {
    "", " ", "-", "_", "na", "n/a", "nan", "none",
    "keine angabe", "keine_angabe", "k.a.", "k. a.",
    "unbekannt", "fehlend", "gefiltert", "filtered"
}

def clean_token(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.lower() in NOINFO:
        return np.nan
    return s

# --- sort helper for mixed numeric/text values ---
def mixed_sort(values):
    # values is a list of unique tokens (strings or numbers); we try numeric sort first
    def to_num(v):
        try:
            return float(v)
        except Exception:
            return None
    nums, texts = [], []
    for v in values:
        n = to_num(v)
        (nums if n is not None else texts).append((n, v))
    if nums and not texts:
        # all numeric-like → sort numerically, keep original string
        return [v for n, v in sorted(nums, key=lambda t: t[0])]
    # otherwise, natural string sort (casefold)
    return sorted(values, key=lambda s: str(s).casefold())

# --- build mapping: Format -> set of possible answers ---
format_to_values = defaultdict(set)
format_to_questions = defaultdict(list)

for _, row in df_meta.iterrows():
    q = row["FrageName"]
    fmt = row["Format"]

    # collect unique cleaned values from the responses for this question
    series = df_resp[q].map(clean_token).dropna()
    uniq = series.unique().tolist()

    for val in uniq:
        format_to_values[fmt].add(val)
    format_to_questions[fmt].append(q)

# --- assemble summary table ---
rows = []
for fmt in sorted(format_to_values.keys(), key=str.casefold):
    vals = mixed_sort(list(format_to_values[fmt]))
    # keep a short preview column plus full list
    preview = ", ".join(vals[:12]) + (" …" if len(vals) > 12 else "")
    rows.append({
        "Format": fmt,
        "Anzahl_Fragen": len(set(format_to_questions[fmt])),
        "Beispiel_Fragen": ", ".join(sorted(set(format_to_questions[fmt]))[:8]) + (
            " …" if len(set(format_to_questions[fmt])) > 8 else ""
        ),
        "Antworten_Liste": " | ".join(vals),
        "Antworten_Vorschau": preview
    })

df_summary = pd.DataFrame(rows).sort_values(["Format"]).reset_index(drop=True)
print(df_summary[["Format", "Anzahl_Fragen", "Antworten_Vorschau"]].to_string(index=False))

# --- save full summary ---
df_summary.to_csv(OUT_PATH, index=False, encoding="utf-8")
print(f"\nSaved: {OUT_PATH}")
