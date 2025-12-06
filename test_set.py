# ---------------------------------------------------------------------
# This script loads LOINC data from a ZIP file, finds records matching
# a clinical query, builds feature vectors, and saves them to a CSV.
#
# Example use:
#   python test_set.py --zip Loinc_2.81.zip --query "cholesterol in plasma"
# ---------------------------------------------------------------------

import argparse
import pandas as pd
import utility_func as utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to the LOINC_2.81.zip file")
    ap.add_argument("--query", required=True, help='Single query, e.g. "cholesterol in plasma"')
    ap.add_argument("--top", type=int, default=15, help="Number of records to select (default: 15)")
    ap.add_argument("--out", default="./outputs/test_dataset.csv", help="Output CSV file path")
    args = ap.parse_args()

    loinc_df = utils.load_loinc_table_from_zip(args.zip)
    subset = utils.pick_candidates(loinc_df, args.query, size = args.top, pos=args.top*0.6)
    subset.insert(0, "qname", args.query.lower())
    feats = utils.build_features(subset, args.query.lower())
    feats.to_csv(args.out, index=False, encoding="utf-8")
    print(f'[OK] File saved to {args.out}')


if __name__ == "__main__":
    main()