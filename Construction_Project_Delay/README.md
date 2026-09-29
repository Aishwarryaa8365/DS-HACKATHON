# Construction Project Delay Analysis

> **Industry:** Construction and Infrastructure  
> **Question:** Why are construction projects getting delayed?

---

## Project Overview

This project analyses 100 construction projects across Residential, Commercial, Infrastructure, and Industrial categories to identify the root causes of project delays. It includes full EDA, delay analysis, and a Random Forest classifier that predicts whether a project will be delayed based on observable project attributes.

---

## Folder Structure

```
Construction_Project_Delay/
├── data/
│   └── project_data.csv              # Raw dataset (100 projects)
├── notebooks/
│   └── project_delay_analysis.ipynb  # Full Jupyter notebook
├── src/
│   ├── data_cleaning.py              # Load, validate, engineer target
│   ├── eda.py                        # EDA + Visualizations 1–3
│   ├── delay_analysis.py             # Delay breakdown by all dimensions
│   ├── random_forest.py              # RF model + Visualization 4
│   └── main.py                       # Pipeline entry point
├── outputs/
│   ├── visualizations/               # 4 PNG charts
│   ├── models/                       # Saved Random Forest model (.pkl)
│   └── reports/                      # CSV reports + model evaluation
├── requirements.txt
├── README.md
└── project_delay_report.md           # Full analysis report
```

---

## Dataset Columns

| Column | Description |
|---|---|
| Project_ID | Unique identifier |
| Project_Category | Residential / Commercial / Infrastructure / Industrial |
| Project_Size | Small / Medium / Large |
| Budget | Project budget (USD) |
| Workforce | Number of workers |
| Material_Delivery | On Time / Delayed |
| Weather | Clear / Rainy / Stormy |
| Contractor | ContractorA / B / C / D |
| Milestone | Foundation / Structural / Finishing |
| Planned_Progress | Expected % completion |
| Actual_Progress | Actual % completion |

---

## Installation

### 1. Clone or navigate to the project folder

```bash
cd Construction_Project_Delay
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run the full pipeline

```bash
python src/main.py
```

This will:
1. Load and clean the dataset
2. Engineer the delay target variable
3. Run EDA and save 3 visualizations
4. Run delay analysis and save CSV reports
5. Train and evaluate the Random Forest model
6. Save the 4th visualization (feature importance)
7. Save the trained model to `outputs/models/`

### Run the Jupyter Notebook

```bash
jupyter notebook notebooks/project_delay_analysis.ipynb
```

---

## Outputs

### Visualizations (`outputs/visualizations/`)
| File | Description |
|---|---|
| `planned_vs_actual_progress.png` | Line chart: Planned vs Actual across all projects |
| `delay_by_project_category.png` | Bar chart: Avg delay gap by category |
| `material_delivery_vs_delay.png` | Box + bar: Material delivery impact on delay |
| `feature_importance.png` | RF feature importance (early warning indicators) |
| `rf_confusion_matrix.png` | Random Forest confusion matrix |

### Reports (`outputs/reports/`)
| File | Description |
|---|---|
| `cleaned_data.csv` | Cleaned dataset with engineered features |
| `delay_by_milestone.csv` | Delay stats per milestone |
| `delay_by_category.csv` | Delay stats per project category |
| `delay_by_contractor.csv` | Delay stats per contractor |
| `delay_by_material_delivery.csv` | Delay stats by material delivery |
| `delay_by_weather.csv` | Delay stats by weather |
| `delay_by_size.csv` | Delay stats by project size |
| `high_risk_projects.csv` | Projects with Progress_Gap ≥ 15% |
| `feature_importance.csv` | RF feature importance scores |
| `model_evaluation.txt` | Accuracy, classification report, confusion matrix |

### Model (`outputs/models/`)
- `random_forest_model.pkl` — Trained RandomForestClassifier with encoders

---

## Key Results

| Metric | Value |
|---|---|
| Dataset size | 100 projects |
| Delay rate | 50% |
| Mean progress gap | 8.84% |
| RF Model Accuracy | **100.00%** |
| Top delay driver | Material Delivery status |

---

## Target Variable Rule

```
Progress_Gap = Planned_Progress − Actual_Progress
Delayed = 1  if  Progress_Gap > 10 pp
Delayed = 0  otherwise
```

---

## Dependencies

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, joblib
- notebook, ipykernel
