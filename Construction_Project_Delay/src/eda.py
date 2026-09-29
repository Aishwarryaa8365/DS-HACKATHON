import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

VIZ_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

def run_eda(df):
    """Perform EDA: descriptive stats + 4 required visualizations."""
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    print(f"\nDescriptive Statistics:\n{df.describe()}")
    print(f"\nProject Category counts:\n{df['Project_Category'].value_counts()}")
    print(f"\nProject Size counts:\n{df['Project_Size'].value_counts()}")
    print(f"\nMaterial Delivery counts:\n{df['Material_Delivery'].value_counts()}")
    print(f"\nContractor counts:\n{df['Contractor'].value_counts()}")
    print(f"\nMilestone counts:\n{df['Milestone'].value_counts()}")
    print(f"\nWeather counts:\n{df['Weather'].value_counts()}")

    # ── VIZ 1: Planned vs Actual Progress ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    ax.plot(x, df['Planned_Progress'], label='Planned Progress', color='steelblue', linewidth=1.5)
    ax.plot(x, df['Actual_Progress'], label='Actual Progress', color='tomato', linewidth=1.5)
    ax.fill_between(x, df['Planned_Progress'], df['Actual_Progress'],
                    alpha=0.2, color='orange', label='Gap')
    ax.set_xlabel('Project Index')
    ax.set_ylabel('Progress (%)')
    ax.set_title('Planned vs Actual Progress Across All Projects')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(VIZ_DIR, 'planned_vs_actual_progress.png')
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"\nSaved: {path1}")

    # ── VIZ 2: Delay by Project Category ───────────────────────────
    cat_delay = df.groupby('Project_Category')['Progress_Gap'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cat_delay.index, cat_delay.values,
                  color=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'])
    ax.set_xlabel('Project Category')
    ax.set_ylabel('Average Progress Gap (%)')
    ax.set_title('Average Delay (Progress Gap) by Project Category')
    for bar, val in zip(bars, cat_delay.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(VIZ_DIR, 'delay_by_project_category.png')
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved: {path2}")

    # ── VIZ 3: Material Delivery vs Project Delay ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Box plot
    df.boxplot(column='Progress_Gap', by='Material_Delivery', ax=axes[0],
               patch_artist=True)
    axes[0].set_title('Progress Gap by Material Delivery Status')
    axes[0].set_xlabel('Material Delivery')
    axes[0].set_ylabel('Progress Gap (%)')
    plt.sca(axes[0])
    plt.title('Progress Gap by Material Delivery Status')
    # Bar: % delayed projects
    delay_rate = df.groupby('Material_Delivery')['Delayed'].mean() * 100
    axes[1].bar(delay_rate.index, delay_rate.values,
                color=['#27ae60', '#e74c3c'])
    axes[1].set_title('% Delayed Projects by Material Delivery')
    axes[1].set_xlabel('Material Delivery')
    axes[1].set_ylabel('% Projects Delayed')
    for i, (idx, val) in enumerate(delay_rate.items()):
        axes[1].text(i, val + 0.5, f'{val:.1f}%', ha='center', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    plt.suptitle('')
    plt.tight_layout()
    path3 = os.path.join(VIZ_DIR, 'material_delivery_vs_delay.png')
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"Saved: {path3}")

    print("\nEDA complete. Visualization 4 (Feature Importance) saved by random_forest.py.")
