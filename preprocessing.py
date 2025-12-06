# ---------------------------------------------------------------------
# This script loads multiple sheets from an Excel dataset, builds
# feature vectors for each query, combines them into one training dataset, 
# and saves it as a CSV file.
#
# Example use:
#   python preprocessing.py --excel loinc_dataset-v2.xlsx
# ---------------------------------------------------------------------

import argparse
from pathlib import Path

import pandas as pd
import utility_func as utils

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=Path, required=True, help="Path to dataset")
    parser.add_argument("--out", type=Path, default=Path("./outputs/training_dataset.csv"), help="Output CSV path")
    args = parser.parse_args()

    # The three sheets we expect
    sheet_names = ["glucose in blood", "bilirubin in plasma", "White blood cells count"]

    frames = []
    for s in sheet_names:
        df = utils.load_sheet_robust(args.excel, s)
        frames.append(utils.build_features(df, s.lower()))

    train_df = pd.concat(frames, ignore_index=True)
    train_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f'[OK] File saved to {args.out}')


if __name__ == "__main__":
    main()