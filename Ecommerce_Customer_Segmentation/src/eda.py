import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "visualizations")

def plot_spend_vs_frequency(df):
    """Viz 1: Scatter — TotalSpend vs ItemsPurchased coloured by MembershipType"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    colors = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}
    labels = {0: 'Bronze', 1: 'Silver', 2: 'Gold'}
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, grp in df.groupby('MembershipType'):
        ax.scatter(grp['ItemsPurchased'], grp['TotalSpend'],
                   color=colors[m], label=labels[m], s=80, edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Items Purchased (Frequency)')
    ax.set_ylabel('Total Spend ($)')
    ax.set_title('Spend vs Frequency by Membership Tier')
    ax.legend(title='Membership')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "viz1_spend_vs_frequency.png"), dpi=150)
    plt.close()

def plot_rfm_heatmap(rfm):
    """Viz 2: Heatmap of avg RFM scores per segment"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    pivot = rfm.groupby('Segment')[['R_Score', 'F_Score', 'M_Score']].mean().astype(float)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap='YlGnBu', linewidths=0.5, ax=ax)
    ax.set_title('Average RFM Scores by Segment')
    ax.set_xlabel('RFM Dimension')
    ax.set_ylabel('Segment')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "viz2_rfm_heatmap.png"), dpi=150)
    plt.close()

def plot_segment_profiles(rfm):
    """Viz 3: Bar chart — avg Recency, Frequency, Monetary per segment (3 subplots)"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    metrics = ['Recency', 'Frequency', 'Monetary']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, color in zip(axes, metrics, colors):
        data = rfm.groupby('Segment')[metric].mean().reindex(['Low', 'Mid', 'High'])
        data.plot(kind='bar', ax=ax, color=color, edgecolor='k', width=0.5)
        ax.set_title(f'Avg {metric} by Segment')
        ax.set_xlabel('Segment')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=0)
    plt.suptitle('Segment Profiles: Recency / Frequency / Monetary', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "viz3_segment_profiles.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_customer_types(df):
    """Viz 4: Quadrant scatter — Frequency vs Spend with quadrant labels"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    freq_med = df['ItemsPurchased'].median()
    spend_med = df['TotalSpend'].median()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['ItemsPurchased'], df['TotalSpend'], color='steelblue', s=80, edgecolors='k', linewidths=0.5)

    for _, row in df.iterrows():
        ax.annotate(row['CustomerID'], (row['ItemsPurchased'], row['TotalSpend']),
                    fontsize=6, ha='center', va='bottom')

    ax.axvline(freq_med, color='gray', linestyle='--', linewidth=1)
    ax.axhline(spend_med, color='gray', linestyle='--', linewidth=1)

    ax.text(freq_med * 0.3, spend_med * 1.6, 'Heavy Spend\nInfrequent', fontsize=8, color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#fadbd8'))
    ax.text(freq_med * 1.3, spend_med * 1.6, 'High-Value\n(Frequent + High Spend)', fontsize=8, color='#1a5276',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#d6eaf8'))
    ax.text(freq_med * 0.3, spend_med * 0.1, 'Low-Value\n(Infrequent + Low Spend)', fontsize=8, color='#7d6608',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#fef9e7'))
    ax.text(freq_med * 1.3, spend_med * 0.1, 'Frequent\nLow Spend', fontsize=8, color='#1e8449',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#d5f5e3'))

    ax.set_xlabel('Items Purchased (Frequency)')
    ax.set_ylabel('Total Spend ($)')
    ax.set_title('Customer Type Quadrant Map')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "viz4_customer_quadrants.png"), dpi=150)
    plt.close()
