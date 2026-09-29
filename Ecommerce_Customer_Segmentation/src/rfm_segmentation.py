import pandas as pd
import os

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "visualizations")
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "reports")

def compute_rfm(df):
    rfm = df[['CustomerID', 'DaysSinceLastPurchase', 'ItemsPurchased', 'TotalSpend']].copy()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])

    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].astype(int).sum(axis=1)
    rfm['Segment'] = pd.cut(rfm['RFM_Score'], bins=[2, 5, 8, 12], labels=['Low', 'Mid', 'High'])
    return rfm

def identify_customer_types(df, rfm):
    """Tag each customer into one of 4 behavioural types based on median splits."""
    merged = df[['CustomerID', 'ItemsPurchased', 'TotalSpend']].merge(
        rfm[['CustomerID', 'Segment', 'RFM_Score']], on='CustomerID')

    freq_med = merged['ItemsPurchased'].median()
    spend_med = merged['TotalSpend'].median()

    def tag(row):
        hi_freq  = row['ItemsPurchased'] >= freq_med
        hi_spend = row['TotalSpend']     >= spend_med
        if hi_freq and hi_spend:
            return 'High-Value'
        elif hi_freq and not hi_spend:
            return 'Frequent Low-Spend'
        elif not hi_freq and hi_spend:
            return 'Heavy-Spend Infrequent'
        else:
            return 'Low-Value'

    merged['CustomerType'] = merged.apply(tag, axis=1)
    return merged

def print_rfm_summary(rfm, customer_types):
    print("\n--- RFM Segment Summary ---")
    summary = rfm.groupby('Segment', observed=True).agg(
        Count=('CustomerID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Avg_RFM_Score=('RFM_Score', 'mean')
    ).round(2)
    print(summary.to_string())

    print("\n--- Customer Type Distribution ---")
    print(customer_types['CustomerType'].value_counts().to_string())

    print("\n--- Customer Type Details ---")
    for ctype, grp in customer_types.groupby('CustomerType'):
        ids = ', '.join(grp['CustomerID'].tolist())
        print(f"  {ctype}: {ids}")

def save_rfm_report(rfm, customer_types):
    os.makedirs(REPORT_DIR, exist_ok=True)
    rfm.to_csv(os.path.join(REPORT_DIR, 'rfm_scores.csv'), index=False)
    customer_types.to_csv(os.path.join(REPORT_DIR, 'customer_types.csv'), index=False)
