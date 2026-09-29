import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from data_cleaning import load_and_clean
from eda import plot_spend_vs_frequency, plot_rfm_heatmap, plot_segment_profiles, plot_customer_types
from rfm_segmentation import compute_rfm, identify_customer_types, print_rfm_summary, save_rfm_report
from knn_model import train_knn

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "customer_data.csv")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")


def generate_insights(df, rfm, customer_types):
    lines = []
    lines.append("=" * 60)
    lines.append("DATASET INSIGHTS")
    lines.append("=" * 60)

    mem_map = {0: 'Bronze', 1: 'Silver', 2: 'Gold'}
    spend_by_mem = df.groupby('MembershipType')['TotalSpend'].mean().round(2)
    lines.append("\n1. Average spend by membership tier:")
    for k, v in spend_by_mem.items():
        lines.append(f"   {mem_map[k]}: ${v:,.2f}")

    rec = rfm.groupby('Segment', observed=True)['Recency'].mean().round(1)
    lines.append("\n2. Average days since last purchase by segment:")
    for seg, val in rec.items():
        lines.append(f"   {seg}: {val} days")

    hv = customer_types[customer_types['CustomerType'] == 'High-Value']
    lines.append(f"\n3. High-Value customers ({len(hv)} total): {', '.join(hv['CustomerID'].tolist())}")
    lines.append(f"   Avg spend: ${hv['TotalSpend'].mean():,.2f} | Avg items: {hv['ItemsPurchased'].mean():.1f}")

    lv = customer_types[customer_types['CustomerType'] == 'Low-Value']
    lines.append(f"\n4. Low-Value customers ({len(lv)} total): {', '.join(lv['CustomerID'].tolist())}")
    lines.append(f"   Avg spend: ${lv['TotalSpend'].mean():,.2f} | Avg items: {lv['ItemsPurchased'].mean():.1f}")

    fl = customer_types[customer_types['CustomerType'] == 'Frequent Low-Spend']
    if not fl.empty:
        lines.append(f"\n5. Frequent Low-Spend customers ({len(fl)} total): {', '.join(fl['CustomerID'].tolist())}")
        lines.append(f"   Buy often but spend below median (${df['TotalSpend'].median():,.2f}).")
        lines.append("   Opportunity: upsell higher-value products.")

    hs = customer_types[customer_types['CustomerType'] == 'Heavy-Spend Infrequent']
    if not hs.empty:
        lines.append(f"\n6. Heavy-Spend Infrequent customers ({len(hs)} total): {', '.join(hs['CustomerID'].tolist())}")
        lines.append("   Spend big but buy rarely.")
        lines.append("   Opportunity: re-engagement campaigns to increase purchase frequency.")

    corr = df['SatisfactionScore'].corr(df['TotalSpend']).round(3)
    lines.append(f"\n7. Correlation between SatisfactionScore and TotalSpend: {corr}")
    if corr > 0.4:
        lines.append("   Strong positive link -- higher satisfaction drives more spending.")
    elif corr > 0.2:
        lines.append("   Moderate positive link -- improving satisfaction may increase spend.")
    else:
        lines.append("   Weak link -- satisfaction alone does not predict spend in this dataset.")

    return "\n".join(lines)


def generate_action_plan(customer_types):
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("MARKETING ACTION PLAN")
    lines.append("=" * 60)

    plan = {
        'High-Value': [
            "Enrol in VIP loyalty programme with exclusive early-access offers.",
            "Send personalised thank-you communications and premium product recommendations.",
            "Offer referral bonuses -- they are most likely to bring similar customers."
        ],
        'Frequent Low-Spend': [
            "Bundle low-cost items into higher-value packs to increase basket size.",
            "Introduce tiered discounts: 'Spend $X more and save Y%'.",
            "Promote premium product lines through targeted email campaigns."
        ],
        'Heavy-Spend Infrequent': [
            "Send re-engagement emails after 30 days of inactivity.",
            "Offer time-limited deals to create urgency and drive repeat visits.",
            "Introduce subscription or auto-replenishment options for preferred categories."
        ],
        'Low-Value': [
            "Run win-back campaigns with first-purchase discount codes.",
            "Send satisfaction surveys to identify and address pain points.",
            "Offer free shipping thresholds to incentivise larger orders."
        ]
    }

    for ctype, actions in plan.items():
        count = len(customer_types[customer_types['CustomerType'] == ctype])
        lines.append(f"\n{ctype} ({count} customers):")
        for i, action in enumerate(actions, 1):
            lines.append(f"  {i}. {action}")

    return "\n".join(lines)


def main():
    print("Loading and cleaning data...")
    df = load_and_clean(DATA_PATH)
    print(f"  {len(df)} customers loaded, {df.shape[1]} features.")

    print("\nComputing RFM segments...")
    rfm = compute_rfm(df)
    customer_types = identify_customer_types(df, rfm)
    print_rfm_summary(rfm, customer_types)

    print("\nGenerating 4 visualizations...")
    plot_spend_vs_frequency(df)
    plot_rfm_heatmap(rfm)
    plot_segment_profiles(rfm)
    plot_customer_types(df)
    print("  Saved to outputs/visualizations/")

    print("\nTraining KNN model + anomaly detection...")
    train_knn(df, rfm)

    print("\nSaving reports...")
    os.makedirs(REPORT_DIR, exist_ok=True)
    save_rfm_report(rfm, customer_types)

    insights    = generate_insights(df, rfm, customer_types)
    action_plan = generate_action_plan(customer_types)
    full_report = insights + "\n" + action_plan

    print("\n" + full_report)

    with open(os.path.join(REPORT_DIR, "insights_and_action_plan.txt"), 'w') as f:
        f.write(full_report)

    print("\nDone! All outputs saved to outputs/")


if __name__ == "__main__":
    main()
