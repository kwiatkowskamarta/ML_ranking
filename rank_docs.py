# ---------------------------------------------------------------------
# This script loads trained SLR models (stage1 and stage2), applies them
# to test data, computes relevance scores for LOINC records, and saves
# ranked results as a CSV file.
#
# Example use:
#   python rank_docs.py --test_csv ./outputs/test_dataset.csv 
# ---------------------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import utility_func as utils

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True, help="Path to test CSV (e.g., slr_test_small.csv)")
    ap.add_argument("--art_dir", default="artifacts", help="Dir with stage1.joblib, stage2.joblib")
    ap.add_argument("--out_csv", default="./outputs/ranked_documents.csv", help="Output ranked CSV")
    ap.add_argument("--topk", type=int, default=10, help="Print top-K per query")
    args = ap.parse_args()

    df = pd.read_csv(args.test_csv)

    # Stage 1
    stage1 = joblib.load(Path(args.art_dir) / "stage1.joblib")
    stage2 = joblib.load(Path(args.art_dir) / "stage2.joblib")

    feat_cols = ['f_component_match', 'f_system_match', 'f_property_match', "f_long_name_match"]
    X = df[feat_cols].astype(float).values

    df['stage1_prob'] = stage1.predict_proba(X)[:, 1]
    df['stage1_logit'] = utils.safe_logit(df['stage1_prob'].values)

    # Calculate Z and N
    key_cols = ["qname", "loinc_num"]
    agg = (
        df.groupby(key_cols, as_index=False)
          .agg(Z=("stage1_logit","sum"),
               N=("stage1_logit","size"))
    )

    # Stage 2
    X2 = agg[['Z', 'N']].astype(float).values
    scaler = joblib.load(Path(args.art_dir) / "stage2_scaler.joblib")
    X2s = scaler.transform(X2)
    agg['final_prob'] = stage2.predict_proba(X2s)[:, 1]

    rep_cols = ['long_common_name', 'component', 'system', 'property']
    rep = (df.groupby(key_cols, as_index=False)[rep_cols].first())
    ranked = agg.merge(rep, on=key_cols, how='left')

    ranked["rank"] = (
        ranked.groupby("qname")["final_prob"]
              .rank(method="first", ascending=False)
              .astype(int)
    )
    ranked = ranked.sort_values(["qname","final_prob"], ascending=[True, False]).reset_index(drop=True)
    ranked = ranked[['rank', 'qname', 'loinc_num', 'long_common_name','component','system','property',
                     'Z', 'N','final_prob']]

    # 8) Save file
    ranked.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved rankings to: {args.out_csv}")

if __name__ == "__main__":
    main()