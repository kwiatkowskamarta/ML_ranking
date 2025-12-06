# ---------------------------------------------------------------------
# This script trains a two-stage logistic regression (SLR) model:
#   - Stage 1: learns local relevance per clue/sample
#   - Stage 2: aggregates and ranks complete LOINC documents
# Models and intermediate outputs are saved to the artifacts directory.
#
# Example use:
#   python model_training.py 
# ---------------------------------------------------------------------
import argparse
import math
from pathlib import Path

import joblib
import pandas as pd
import utility_func as utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("./outputs/training_dataset.csv"),
                        help="Path to the training CSV produced earlier.")
    parser.add_argument("--outdir", type=Path, default=Path("./artifacts"),
                        help="Directory to save models and predictions.")
    parser.add_argument("--test_size", type=float, default=0.25,
                        help="Holdout fraction for evaluation.")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Read data
    df = pd.read_csv(args.data)

    key_cols = ["qname", "loinc_num"]

    # Stage-1 features (from data-prep)
    feat_cols = ["f_component_match","f_system_match","f_property_match","f_long_name_match"]

    X = df[feat_cols].astype(float).values
    y = df["label"].astype(int).values

    # Split train/valid
    X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(
        X, y, df.index.values, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Stage-1: Logistic Regression
    stage1 = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=200,
        random_state=args.random_state
    )
    stage1.fit(X_train, y_train)

    # Evaluate Stage-1 on validation
    prob_valid = stage1.predict_proba(X_valid)[:, 1]
    logit_valid = utils.safe_logit(prob_valid)
    pred_valid = (prob_valid >= 0.5).astype(int)
    utils.print_metrics(y_valid, prob_valid, pred_valid, header="Stage-1 (per clue/sample)")

     # Save Stage-1 model
    joblib.dump(stage1, args.outdir / "stage1.joblib")
    print(f"[OK] Saved Stage-1 model to {args.outdir / 'stage1.joblib'}")

    # Add predictions/logits back to full dataframe
    df_stage1 = df.copy()
    df_stage1["stage1_prob"] = stage1.predict_proba(X)[:, 1]
    df_stage1["stage1_logit"] = utils.safe_logit(df_stage1["stage1_prob"].values)
    df_stage1.to_csv(args.outdir / "stage1_predictions.csv", index=False)
    print(f"[OK] Saved per-sample Stage-1 predictions to {args.outdir / 'stage1_predictions.csv'}")
    
    print('--------------------------------')
    print(df_stage1)
    df_stage1.to_csv('./outputs/stage1.csv')
    print('--------------------------------')

    # Stage-2
    agg = (
        df_stage1
        .groupby(key_cols)
        .agg(
            Z=("stage1_logit", "sum"),
            N=("stage1_logit", "size"),
            label=("label", "max")
        )
        .reset_index()
    )

    # print(agg)

    # Prepare Stage-2 features
    X2 = agg[["Z", "N"]].astype(float).values
    y2 = agg["label"].astype(int).values

    # Split stage-2 train/valid
    X2_train, X2_valid, y2_train, y2_valid = train_test_split(
        X2, y2, test_size=args.test_size, random_state=args.random_state, stratify=y2
    )

    scaler = StandardScaler()
    X2_train_s = scaler.fit_transform(X2_train)
    X2_valid_s = scaler.transform(X2_valid)

    stage2 = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=200,
        random_state=args.random_state
    )
    stage2.fit(X2_train_s, y2_train)

    # Evaluate Stage-2
    prob2_valid = stage2.predict_proba(X2_valid_s)[:, 1]
    pred2_valid = (prob2_valid >= 0.5).astype(int)
    utils.print_metrics(y2_valid, prob2_valid, pred2_valid, header="Stage-2 (per document)")

    # Save Stage-2 model 
    joblib.dump(stage2, args.outdir / "stage2.joblib")
    joblib.dump(scaler, args.outdir / "stage2_scaler.joblib")
    print(f"[OK] Saved Stage-2 model to {args.outdir / 'stage2.joblib'}")

    X2_all_s = scaler.transform(X2)
    agg["stage2_prob"] = stage2.predict_proba(X2_all_s)[:, 1]
    agg.to_csv(args.outdir / "stage2_predictions.csv", index=False)
    print(f"[OK] Saved per-(Q,D) Stage-2 predictions to {args.outdir / 'stage2_predictions.csv'}")

    # Small sanity print
    print("\nSample Stage-2 rows:")
    print(agg.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
