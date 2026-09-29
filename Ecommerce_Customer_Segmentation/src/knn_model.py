import pickle, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE, "outputs", "models")
VIZ_DIR    = os.path.join(BASE, "outputs", "visualizations")
REPORT_DIR = os.path.join(BASE, "outputs", "reports")

FEATURES = ['Age', 'TotalSpend', 'ItemsPurchased', 'DaysSinceLastPurchase',
            'AveragePurchaseValue', 'SatisfactionScore', 'MembershipType']

def train_knn(df, rfm):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    merged = df.merge(rfm[['CustomerID', 'Segment']], on='CustomerID')
    merged.dropna(subset=['Segment'], inplace=True)

    le = LabelEncoder()
    y = le.fit_transform(merged['Segment'])
    X = merged[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
        stratify=y if len(set(y)) > 1 else None)

    # Best K via cross-validation
    max_k = min(10, len(X_train) - 1)
    k_scores = []
    for k in range(1, max_k + 1):
        cv = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_scaled, y,
                             cv=min(5, len(X_scaled)), scoring='accuracy')
        k_scores.append(cv.mean())
    best_k = int(np.argmax(k_scores)) + 1

    # Elbow plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, max_k + 1), k_scores, marker='o', color='steelblue')
    plt.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
    plt.title('KNN: Accuracy vs K (Cross-Validation)')
    plt.xlabel('K')
    plt.ylabel('CV Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "knn_elbow.png"), dpi=150)
    plt.close()

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    present = sorted(set(y_test) | set(y_pred))
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=present,
                                   target_names=le.classes_[present])

    print(f"\n--- KNN Classifier (Best K={best_k}) ---")
    print(f"Test Accuracy: {acc:.2%}")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=present)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=le.classes_[present],
                yticklabels=le.classes_[present],
                cmap='Blues')
    plt.title(f'KNN Confusion Matrix (K={best_k}, Acc={acc:.0%})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "knn_confusion_matrix.png"), dpi=150)
    plt.close()

    # KNN Anomaly Detection
    anomalies = detect_anomalies(X_scaled, merged, le, y)

    with open(os.path.join(MODEL_DIR, "knn_model.pkl"), 'wb') as f:
        pickle.dump({'model': knn, 'scaler': scaler, 'encoder': le,
                     'best_k': best_k, 'features': FEATURES}, f)

    with open(os.path.join(REPORT_DIR, "knn_evaluation.txt"), 'w') as f:
        f.write(f"Best K: {best_k}\n")
        f.write(f"Test Accuracy: {acc:.2%}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nAnomalies Detected:\n")
        f.write(anomalies.to_string(index=False) if not anomalies.empty else "None")

    return knn, scaler, le


def detect_anomalies(X_scaled, merged, le, y):
    """Flag customers whose nearest neighbours are mostly from a different segment."""
    k = min(5, len(X_scaled) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)

    rows = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        neighbour_labels = y[idx[1:]]
        own_label = y[i]
        mismatch_rate = np.mean(neighbour_labels != own_label)
        avg_dist = dist[1:].mean()
        if mismatch_rate >= 0.6:
            rows.append({
                'CustomerID': merged.iloc[i]['CustomerID'],
                'Segment': le.classes_[own_label],
                'Mismatch_Rate': round(mismatch_rate, 2),
                'Avg_NN_Distance': round(avg_dist, 4)
            })

    anomalies = pd.DataFrame(rows)
    print("\n--- Anomaly Detection ---")
    if anomalies.empty:
        print("No strong anomalies detected.")
    else:
        print(anomalies.to_string(index=False))
    return anomalies
