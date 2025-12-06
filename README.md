# ML_ranking: Ranking Clinical Queries and LOINC Tests using Staged Logistic Regression

**Authors:** Marta Kwiatkowska, Weronika Pędzimąż  
**Institution:** Universidad Politécnica de Madrid  
**Date:** October 2025

## Project Overview
This project develops an automated system for ranking the relevance between clinical text queries and structured laboratory test metadata (LOINC). Based on the Staged Logistic Regression (SLR) approach, it uses a two-stage logistic model to estimate how well each test matches a given query.

The system automatically extracts features from Excel datasets, trains logistic models, and links clinical queries with the most relevant standardized test descriptions.

## Key Features
- **Automated Preprocessing:** Unifies data from multiple Excel sheets without manual labeling.
- **Two-Stage Modeling:** 1. **Stage 1:** Predicts likelihood of relevance for individual records.
  2. **Stage 2:** Aggregates results to rank the entire record set.
- **Test Set Generation:** Automatically builds test datasets from the official LOINC archive.
- **Modular Design:** Independent scripts for preprocessing, training, testing, and ranking.

## Project Structure
```text
- code/
  ├── artifacts/          # Saved models and scalers
  ├── outputs/            # Generated datasets and results
  ├── loinc_dataset-v2/   # Input data
  ├── model_training/     # Logic for training SLR models
  ├── preprocessing/      # Data cleaning and feature extraction
  ├── rank_docs/          # Ranking script for new queries
  ├── test_set/           # Test set generation from LOINC archive
  ├── utility_func/       # Helper functions (metrics, loading data)
  ├── requirements.txt    # Python dependencies
  └── README.md           # Project documentation
