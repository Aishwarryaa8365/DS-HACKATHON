"""
Construction Project Delay Analysis
====================================
Run the full pipeline:
    python src/main.py
"""
import os
import sys

# Allow imports from src/ regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from data_cleaning  import load_and_clean_data
from eda            import run_eda
from delay_analysis import analyze_delays
from random_forest  import train_and_evaluate

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'project_data.csv')

def main():
    print("=" * 60)
    print("  CONSTRUCTION PROJECT DELAY — FULL ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1 — Load & Clean
    df = load_and_clean_data(DATA_PATH)

    # Step 2 — EDA + Visualizations 1-3
    run_eda(df)

    # Step 3 — Delay Analysis
    analyze_delays(df)

    # Step 4 — Random Forest + Visualization 4
    rf, acc, report, cm, importances = train_and_evaluate(df)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Model Accuracy : {acc*100:.2f}%")
    print("  All outputs saved in outputs/")
    print("=" * 60)

if __name__ == '__main__':
    main()
