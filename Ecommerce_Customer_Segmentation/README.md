# Ecommerce Customer Segmentation

Segments ecommerce customers using **RFM Analysis** and **KNN Classification**.

## Project Structure
```
Ecommerce_Customer_Segmentation/
├── data/               # Raw customer data
├── notebooks/          # Jupyter notebook for exploration
├── src/                # Source modules
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── rfm_segmentation.py
│   ├── knn_model.py
│   └── main.py
└── outputs/            # Visualizations, models, reports
```

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
cd src
python main.py
```

## Approach
1. **Data Cleaning** — Remove duplicates, nulls, encode categoricals
2. **EDA** — Distribution plots, correlation heatmap
3. **RFM Segmentation** — Score customers on Recency, Frequency, Monetary → Low / Mid / High
4. **KNN Model** — Classify new customers into segments using k-nearest neighbors

## Outputs
- `outputs/visualizations/` — All plots
- `outputs/models/knn_model.pkl` — Trained model
- `outputs/reports/rfm_scores.csv` — RFM scores per customer
