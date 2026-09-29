import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

VIZ_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'visualizations')
MDL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')
RPT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports')
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)

CATEGORICAL_COLS = ['Project_Category', 'Project_Size', 'Material_Delivery',
                    'Weather', 'Contractor', 'Milestone']
NUMERICAL_COLS   = ['Budget', 'Workforce', 'Planned_Progress']
TARGET           = 'Delayed'

def preprocess(df):
    """Encode categoricals and select features."""
    df_enc = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col + '_enc'] = le.fit_transform(df_enc[col])
        encoders[col] = le
    feature_cols = NUMERICAL_COLS + [c + '_enc' for c in CATEGORICAL_COLS]
    X = df_enc[feature_cols]
    y = df_enc[TARGET]
    return X, y, encoders, feature_cols

def train_and_evaluate(df):
    """Train Random Forest, evaluate, save model and visualizations."""
    print("\n=== RANDOM FOREST MODEL ===")

    X, y, encoders, feature_cols = preprocess(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")

    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['On Track', 'Delayed'])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"\nClassification Report:\n{report}")
    print(f"\nConfusion Matrix:\n{cm}")

    # Save metrics to report
    with open(os.path.join(RPT_DIR, 'model_evaluation.txt'), 'w') as f:
        f.write(f"Random Forest Model Evaluation\n{'='*40}\n")
        f.write(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # ── Confusion Matrix plot ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['On Track', 'Delayed'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Random Forest — Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(VIZ_DIR, 'rf_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved: {cm_path}")

    # ── VIZ 4: Feature Importance ───────────────────────────────────
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    # Rename for readability
    rename_map = {
        'Budget': 'Budget',
        'Workforce': 'Workforce',
        'Planned_Progress': 'Planned Progress',
        'Project_Category_enc': 'Project Category',
        'Project_Size_enc': 'Project Size',
        'Material_Delivery_enc': 'Material Delivery',
        'Weather_enc': 'Weather',
        'Contractor_enc': 'Contractor',
        'Milestone_enc': 'Milestone'
    }
    importances.index = [rename_map.get(i, i) for i in importances.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#e74c3c' if v >= importances.quantile(0.75) else '#3498db'
              for v in importances.values]
    importances.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Random Forest — Feature Importance\n(Early Warning Indicators)', fontsize=13)
    ax.set_xlabel('Importance Score')
    ax.axvline(importances.mean(), color='gray', linestyle='--', label='Mean importance')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fi_path = os.path.join(VIZ_DIR, 'feature_importance.png')
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Saved: {fi_path}")

    # Save feature importance CSV
    fi_df = importances.sort_values(ascending=False).reset_index()
    fi_df.columns = ['Feature', 'Importance']
    fi_df.to_csv(os.path.join(RPT_DIR, 'feature_importance.csv'), index=False)

    # Save model
    model_path = os.path.join(MDL_DIR, 'random_forest_model.pkl')
    joblib.dump({'model': rf, 'encoders': encoders, 'features': feature_cols}, model_path)
    print(f"Model saved: {model_path}")

    return rf, acc, report, cm, importances
