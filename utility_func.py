import re
import io
import zipfile
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score
)

# Configuration
# Predefined matching rules for known queries
QUERY_RULES = {
    "glucose in blood": {
        "components": ["glucose"],
        "systems_positive": ["bld", "whole blood", "wb", "blood"],
        "properties": ["scnc", "mcnc", "acnc"],
    },
    "bilirubin in plasma": {
        "components": [
            "bilirubin", "bilirubin.total", "total bilirubin",
            "conjugated bilirubin", "unconjugated bilirubin"
        ],
        "systems_positive": ["ser/plas", "plasma", "serum"],
        "properties": ["scnc", "mcnc", "acnc", "prthr"],
    },
    "white blood cells count": {
        "components": [
            "wbc", "white blood cell", "white blood cells",
            "leukocyte", "leukocytes", "leucocyte", "leucocytes"
        ],
        "systems_positive": ["bld", "whole blood", "wb", "blood"],
        "properties": ["num", "ncnt", "ncnc", "#", "cnt"],
    },
}

EXPECTED_COLS = ["LOINC_NUM", "LONG_COMMON_NAME", "COMPONENT", "SYSTEM", "PROPERTY"]
LIKELY_TABLE_HINTS = ["loinc.csv", "loinccore", "loinctable", "loinctablecore", ".csv"]

# Text normalization and matching helpers

def normalize_token(s: str) -> str:
    """Lowercase text, trim spaces, collapse multiple spaces."""
    return re.sub(r"\s+", " ", str(s).lower().strip())

def contains_any(text: str, needles) -> bool:
    """Check if any of the given tokens appears in the text."""
    t = normalize_token(text)
    return any(n in t for n in needles)

def token_in(text: str, tokens) -> int:
    """Return 1 if any token is present in text, else 0."""
    return int(contains_any(text, tokens))

def system_match(text: str, system_rules) -> int:
    """Return 1 if text matches or contains any system keyword."""
    t = normalize_token(text)
    return int(any(t == s or s in t for s in system_rules))

def property_match(prop: str, allowed) -> int:
    """Return 1 if property field contains any allowed keyword."""
    p = normalize_token(prop)
    return int(any(tok in p for tok in allowed))

def long_name_match(long_name: str, query: str) -> int:
    """Return 1 if any query token appears in LOINC long common name."""
    q = normalize_token(query)
    q_tokens = [t for t in re.split(r"[^a-z0-9+]+", q) if len(t) > 2]
    name = normalize_token(long_name)
    return int(any(tok in name for tok in q_tokens))

def safe_logit(p, eps: float = 1e-9):
    """Compute stable logit = log(p/(1-p)) with clipping."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

# Loading Excel or ZIP data 

def load_sheet_robust(path: str, sheet_name: str) -> pd.DataFrame:
    """Load Excel sheet and detect header row containing 'loinc_num'."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row_idx = None
    for i in range(min(10, len(raw))):
        if (raw.iloc[i].astype(str).str.lower() == 'loinc_num').any():
            header_row_idx = i
            break
    if header_row_idx is None:
        header_row_idx = 1
    header = raw.iloc[header_row_idx].astype(str).str.strip().str.lower().tolist()
    df = raw[header_row_idx + 1:].copy()
    df.columns = header
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def _read_csv_from_zip(z: zipfile.ZipFile, name: str) -> pd.DataFrame | None:
    """Try reading a CSV inside ZIP with fallback to Latin-1 encoding."""
    try:
        with z.open(name) as fh:
            try:
                return pd.read_csv(
                    fh, dtype=str, keep_default_na=False,
                    usecols=lambda c: c.upper() in EXPECTED_COLS
                )
            except UnicodeDecodeError:
                fh2 = z.open(name)
                return pd.read_csv(
                    io.TextIOWrapper(fh2, encoding='latin-1'),
                    dtype=str, keep_default_na=False,
                    usecols=lambda c: c.upper() in EXPECTED_COLS
                )
    except Exception:
        return None

def load_loinc_table_from_zip(zip_path: str,
                              expected_cols=EXPECTED_COLS,
                              likely_hints=LIKELY_TABLE_HINTS) -> pd.DataFrame:
    """Find and load main LOINC table CSV from the ZIP archive."""
    with zipfile.ZipFile(zip_path) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        csvs = sorted(csvs, key=lambda n: sum(h in n.lower() for h in likely_hints), reverse=True)
        for n in csvs:
            df = _read_csv_from_zip(z, n)
            if df is None:
                continue
            cols_up = {c.upper() for c in df.columns}
            if all(c in cols_up for c in expected_cols):
                df = df[[c for c in df.columns if c.upper() in expected_cols]].copy()
                df.rename(columns={c: c.upper() for c in df.columns}, inplace=True)
                for c in expected_cols:
                    df[c] = df[c].astype(str).str.strip()
                return df
    raise FileNotFoundError("No valid LOINC table found in ZIP.")

# Query-based rule derivation and candidate selection

def derive_rules_from_query(q: str) -> dict:
    """Map query words to predefined component, system, and property patterns."""
    qn = normalize_token(q)
    comp_tokens, sys_pos = [], []
    if " in " in qn:
        left, right = qn.split(" in ", 1)
        comp_tokens = [left.strip()]
        if "blood" in right:
            sys_pos = ["bld", "whole blood", "wb", "blood"]
        elif "plasma" in right:
            sys_pos = ["ser/plas", "plasma"]
        elif "serum" in right:
            sys_pos = ["ser/plas", "serum"]
        elif "urine" in right:
            sys_pos = ["urine", "urn"]
        else:
            sys_pos = [right.strip()]
    else:
        comp_tokens = [qn]
        sys_pos = []
    if any(tok in qn for tok in ["count", "number", "cells", "wbc", "neutrophils", "lymphocytes", "#"]):
        prop_tokens = ["num", "ncnt", "ncnc", "#", "cnt"]
    else:
        prop_tokens = ["scnc", "mcnc", "acnc"]

    synonyms = {
        "wbc": ["white blood cell", "white blood cells", "leukocyte", "leukocytes", "leucocyte", "leucocytes", "wbc"],
        "glucose": ["glucose", "blood sugar", "dextrose"],
        "bilirubin": ["bilirubin", "bilirubin total", "total bilirubin",
                      "conjugated bilirubin", "unconjugated bilirubin"],
        "cholesterol": ["cholesterol", "ldl cholesterol", "hdl cholesterol", "total cholesterol"],
        "hemoglobin": ["hemoglobin", "haemoglobin", "hgb"],
        "potassium": ["potassium", "k+"],
        "sodium": ["sodium", "na+"],
    }
    for k, vals in synonyms.items():
        if k in qn:
            comp_tokens = list(set(comp_tokens + vals))

    return {
        "components": comp_tokens,
        "systems_positive": sys_pos,
        "properties": prop_tokens,
    }

def pick_candidates(loinc_df: pd.DataFrame, query: str, size: int, pos: int) -> pd.DataFrame:
    """Return a mix of relevant and irrelevant LOINC records for a given query."""
    qn = normalize_token(query)
    rules = derive_rules_from_query(qn)
    comp = [normalize_token(x) for x in rules["components"]]
    sys = [normalize_token(x) for x in rules["systems_positive"]]
    prop = [normalize_token(x) for x in rules["properties"]]

    # simple scoring
    def score_row(row):
        score = 0
        if any(w in normalize_token(row["COMPONENT"]) for w in comp):
            score += 1
        if any(w in normalize_token(row["SYSTEM"]) for w in sys):
            score += 1
        if any(w in normalize_token(row["PROPERTY"]) for w in prop):
            score += 1
        return score

    loinc_df = loinc_df.copy()
    loinc_df["score"] = loinc_df.apply(score_row, axis=1)

    # sampling 
    matched = loinc_df[loinc_df["score"] > 0].sort_values("score", ascending=False)
    nonmatched = loinc_df[loinc_df["score"] == 0]
    pos = int(pos)
    size = int(size)
    neg = max(size - pos, 0)

    top_pos = matched.head(pos)
    top_neg = (
        nonmatched.sample(n=neg, random_state=42)
        if len(nonmatched) >= neg and neg > 0
        else nonmatched
    )

    result = pd.concat([top_pos, top_neg], ignore_index=True)
    result = result.sample(frac=1.0, random_state=42).reset_index(drop=True)
    result.drop(columns=["score"], inplace=True, errors="ignore")
    return result

# Feature building for SLR

def get_rules(qname: str) -> dict:
    """Return predefined or dynamically derived rules for a query."""
    q = normalize_token(qname)
    if q not in QUERY_RULES:
        QUERY_RULES[q] = derive_rules_from_query(q)
    return QUERY_RULES[q]

def build_features(df: pd.DataFrame, qname: str) -> pd.DataFrame:
    """Compute binary features for the SLR model."""
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    q = normalize_token(qname)
    out["qname"] = q

    rules = get_rules(q)
    out["f_component_match"] = out["component"].apply(lambda x: token_in(x, rules["components"]))
    out["f_system_match"] = out["system"].apply(lambda x: system_match(x, rules["systems_positive"]))
    out["f_property_match"] = out["property"].apply(lambda x: property_match(x, rules["properties"]))
    out["f_long_name_match"] = out["long_common_name"].apply(lambda x: long_name_match(x, qname))

    out["label"] = (
        out["f_component_match"]
        & out["f_system_match"]
        & out["f_property_match"]
    ).astype(int)
    return out

# Metrics printing

def print_metrics(y_true, y_prob, y_pred, header: str):
    """Print standard classification metrics."""
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, y_prob)
    except ValueError:
        pr = float("nan")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n=== {header} ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1:          {f1:.4f}")
    print(f"ROC-AUC:     {roc:.4f}")
    print(f"PR-AUC:      {pr:.4f}")
