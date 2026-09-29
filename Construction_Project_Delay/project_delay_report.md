# Construction Project Delay Report

---

## 1. Problem Statement

Construction projects across Residential, Commercial, Infrastructure, and Industrial sectors frequently suffer from delays that lead to cost overruns, contractual penalties, and resource wastage. This project investigates **why construction projects get delayed** by analysing planned vs actual progress, identifying the operational factors that drive delays, and building a machine learning model to predict which projects are at risk.

---

## 2. Objectives

- Clean and validate the construction project dataset.
- Perform exploratory data analysis (EDA) and descriptive statistics.
- Calculate and visualise the gap between planned and actual progress.
- Identify milestones and project categories where delays accumulate most.
- Analyse the relationship between material delivery status and project delay.
- Compare delay patterns across project categories and contractors.
- Identify early warning indicators of project delay.
- Build and evaluate a Random Forest classifier to predict project delay.
- Provide a practical action plan for construction management.

---

## 3. Dataset Description

| Column | Type | Description |
|---|---|---|
| Project_ID | String | Unique project identifier (P001–P100) |
| Project_Category | Categorical | Residential, Commercial, Infrastructure, Industrial |
| Project_Size | Categorical | Small, Medium, Large |
| Budget | Numeric | Project budget in USD |
| Workforce | Numeric | Number of workers assigned |
| Material_Delivery | Categorical | On Time / Delayed |
| Weather | Categorical | Clear / Rainy / Stormy |
| Contractor | Categorical | ContractorA / B / C / D |
| Milestone | Categorical | Foundation / Structural / Finishing |
| Planned_Progress | Numeric | Expected % completion at snapshot |
| Actual_Progress | Numeric | Actual % completion at snapshot |

- **Total records:** 100 projects
- **No missing values** detected.
- **No duplicate rows** detected.

---

## 4. Data Cleaning

- Loaded 100 rows × 11 columns.
- Stripped whitespace from all string columns.
- Validated that Planned_Progress and Actual_Progress are within [0, 100].
- Validated that Budget and Workforce are positive.
- **No rows were dropped** — the dataset was already clean.

### Target Engineering Rule (documented)

Since the dataset does not contain an explicit delay label, a data-driven target was created:

```
Progress_Gap = Planned_Progress − Actual_Progress

Delayed = 1  if  Progress_Gap > 10 percentage points
Delayed = 0  otherwise  (On Track)
```

This threshold of 10 pp represents a meaningful operational lag — a project that is more than 10 percentage points behind its planned schedule is considered delayed. The resulting class distribution is **perfectly balanced: 50 Delayed, 50 On Track**.

---

## 5. EDA — Descriptive Statistics

| Metric | Value |
|---|---|
| Mean Budget | $4,130,400 |
| Mean Workforce | 107.8 workers |
| Mean Planned Progress | 54.04% |
| Mean Actual Progress | 45.20% |
| Mean Progress Gap | 8.84% |
| Std Dev Progress Gap | 7.01% |
| Min Progress Gap | 1% |
| Max Progress Gap | 20% |

**Category distribution:** 25 projects each across Residential, Commercial, Infrastructure, Industrial.  
**Contractor distribution:** 25 projects each across ContractorA, B, C, D.  
**Material Delivery:** 50 On Time, 50 Delayed.  
**Weather:** 50 Clear, 25 Rainy, 25 Stormy.  
**Milestone:** 34 Structural, 33 Foundation, 33 Finishing.

---

## 6. Planned vs Actual Progress

- Overall mean planned progress: **54.04%**
- Overall mean actual progress: **45.20%**
- Overall mean progress gap: **8.84 percentage points**
- **50 out of 100 projects (50%)** are classified as Delayed (gap > 10 pp).

The planned vs actual progress chart (Visualization 1) shows a consistent gap across the project portfolio, with the gap widening noticeably for Industrial and Commercial projects.

---

## 7. Delay Analysis

### By Milestone

| Milestone | Avg Gap (%) | Delay Rate |
|---|---|---|
| Finishing | 9.73 | 48% |
| Structural | 9.29 | 53% |
| Foundation | 7.48 | 48% |

Structural milestones show the highest delay rate (53%), indicating that structural work is the most vulnerable phase.

### By Project Size

| Size | Avg Gap (%) | Delay Rate |
|---|---|---|
| Large | 10.44 | 59% |
| Medium | 8.91 | 53% |
| Small | 6.62 | 34% |

Larger projects carry significantly higher delay risk.

---

## 8. Material Delivery Analysis

| Material Delivery | Avg Gap (%) | Delay Rate |
|---|---|---|
| Delayed | 15.66 | 100% |
| On Time | 2.02 | 0% |

**Material delivery status is the single strongest predictor of project delay.** Every project with delayed material delivery was classified as Delayed, while every project with on-time delivery was On Track. This is a critical operational finding.

---

## 9. Project Category Comparison

| Category | Avg Gap (%) | Delay Rate | Avg Budget |
|---|---|---|---|
| Industrial | 16.64 | 100% | $5,956,000 |
| Commercial | 14.68 | 100% | $4,062,000 |
| Infrastructure | 2.12 | 0% | $5,798,000 |
| Residential | 1.92 | 0% | $705,600 |

Industrial and Commercial projects are universally delayed in this dataset, while Infrastructure and Residential projects are consistently on track. This aligns with the contractor assignment pattern — ContractorD handles Industrial and ContractorB handles Commercial projects, both of which operate under Stormy/Rainy weather conditions with delayed material delivery.

### Contractor Comparison

| Contractor | Avg Gap (%) | Delay Rate |
|---|---|---|
| ContractorD | 16.64 | 100% |
| ContractorB | 14.68 | 100% |
| ContractorC | 2.12 | 0% |
| ContractorA | 1.92 | 0% |

ContractorD and ContractorB have 100% delay rates. ContractorA and ContractorC have 0% delay rates.

---

## 10. Early Warning Indicators

The following signals, detectable early in a project lifecycle, are strongly associated with eventual delay:

1. **Material Delivery = Delayed** — 100% delay rate; the most powerful single indicator.
2. **Weather = Stormy or Rainy** — All stormy/rainy projects are delayed; clear weather projects are on track.
3. **Project Category = Industrial or Commercial** — 100% delay rate in this dataset.
4. **Contractor = ContractorD or ContractorB** — 100% delay rate; contractor selection is a leading indicator.
5. **Project Size = Large** — 59% delay rate vs 34% for Small projects.
6. **Milestone = Structural** — Highest delay rate (53%) among all milestone stages.
7. **High Planned Progress with Low Actual Progress** — A widening gap at any early milestone is a direct warning sign.

---

## 11. Random Forest Methodology

### Feature Engineering
- Categorical features (Project_Category, Project_Size, Material_Delivery, Weather, Contractor, Milestone) were encoded using LabelEncoder.
- Numerical features (Budget, Workforce, Planned_Progress) were used as-is.
- Total features: 9

### Model Configuration
- Algorithm: RandomForestClassifier
- n_estimators: 100
- random_state: 42
- class_weight: 'balanced'
- Train/Test split: 80/20 (stratified)

### Target Variable
- Binary classification: Delayed (1) vs On Track (0)
- Derived from Progress_Gap > 10 pp rule

---

## 12. Model Evaluation

| Metric | On Track | Delayed | Overall |
|---|---|---|---|
| Precision | 1.00 | 1.00 | — |
| Recall | 1.00 | 1.00 | — |
| F1-Score | 1.00 | 1.00 | — |
| Support | 10 | 10 | 20 |
| **Accuracy** | — | — | **100.00%** |

### Confusion Matrix

```
              Predicted
              On Track  Delayed
Actual On Track   10        0
       Delayed     0       10
```

The model achieves perfect accuracy on the test set. This is expected given the strong, clean separation between delayed and on-track projects driven by Material_Delivery and Weather — both of which are perfectly correlated with the delay label in this dataset.

---

## 13. Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | Material Delivery | Highest |
| 2 | Weather | High |
| 3 | Contractor | High |
| 4 | Project Category | High |
| 5 | Planned Progress | Moderate |
| 6 | Budget | Moderate |
| 7 | Workforce | Moderate |
| 8 | Project Size | Low–Moderate |
| 9 | Milestone | Low |

Material Delivery and Weather dominate feature importance, confirming that external supply chain and environmental factors are the primary drivers of construction project delay.

---

## 14. Key Insights (5–7)

1. **Material delivery is the #1 delay driver.** All 50 projects with delayed material delivery were classified as Delayed (avg gap: 15.66%), while all 50 on-time delivery projects were On Track (avg gap: 2.02%). Fixing supply chain reliability would eliminate 100% of delays in this dataset.

2. **Weather conditions are a decisive factor.** Stormy and Rainy weather projects have 100% delay rates with average gaps of 16.64% and 14.68% respectively. Clear weather projects average only a 2.02% gap. Weather-adaptive scheduling is essential.

3. **Industrial projects are the highest-risk category.** With a 100% delay rate and an average progress gap of 16.64%, Industrial projects consistently underperform. They are also the highest-budget category (avg $5.96M), making delays extremely costly.

4. **Contractor performance is highly polarised.** ContractorA and ContractorC deliver 0% delay rates, while ContractorB and ContractorD deliver 100% delay rates. Contractor selection is a controllable, high-impact decision.

5. **Large projects are 1.7× more likely to be delayed than small ones.** Large projects have a 59% delay rate vs 34% for small projects, suggesting that project complexity and scale amplify delay risk.

6. **Structural milestones are the most vulnerable phase.** With a 53% delay rate, structural work is where delays most frequently crystallise. Early intervention at this stage can prevent cascading delays.

7. **The Random Forest model achieves 100% accuracy**, confirming that the delay pattern in this dataset is fully explained by the combination of Material_Delivery, Weather, and Contractor — all of which are observable before or at project start.

---

## 15. Operational Causes of Delays

| Cause | Impact Level | Evidence |
|---|---|---|
| Delayed material delivery | Critical | 100% delay rate, avg gap 15.66% |
| Adverse weather (Stormy/Rainy) | Critical | 100% delay rate for both conditions |
| Contractor capability gaps | High | ContractorB/D: 100% delay; A/C: 0% |
| Large project complexity | Medium | Large: 59% delay vs Small: 34% |
| Structural phase challenges | Medium | Highest milestone delay rate (53%) |
| Industrial project characteristics | High | 100% delay, highest avg gap (16.64%) |

---

## 16. Practical Action Plan for Construction Management

### Immediate Actions (0–30 days)
1. **Audit all material supply chains** for active projects. Flag any supplier with a history of late delivery and establish backup suppliers for critical materials.
2. **Review ContractorB and ContractorD contracts** — investigate root causes of their 100% delay rates and implement performance improvement plans or consider retendering.

### Short-Term Actions (1–3 months)
3. **Implement weather-contingency scheduling** for all projects in Rainy/Stormy-prone regions. Build buffer time into Structural milestones specifically.
4. **Deploy early warning dashboards** that flag projects the moment Material_Delivery status changes to "Delayed" — this is the single most predictive signal.
5. **Apply the Random Forest model** at project initiation to score delay risk based on contractor, weather forecast, material delivery status, and project size.

### Long-Term Actions (3–12 months)
6. **Standardise contractor performance metrics** — track delay rates per contractor across all projects and use this data in future procurement decisions.
7. **Invest in prefabrication and modular construction** for Industrial and Commercial projects to reduce dependency on on-site material delivery timelines.
8. **Establish milestone-based progress reviews** at Foundation and Structural stages with mandatory corrective action triggers if actual progress falls more than 5 pp behind planned.

---

## 17. Conclusion

This analysis of 100 construction projects reveals that project delays are not random — they are systematically driven by a small set of controllable and observable factors. **Material delivery status and weather conditions together explain virtually all delay variance** in this dataset, with contractor selection as a closely related third factor.

The Random Forest model confirms these findings with 100% classification accuracy, demonstrating that delay risk can be predicted reliably at project initiation using information that is available before construction begins.

The most impactful intervention is supply chain management: ensuring on-time material delivery would, based on this data, eliminate all classified delays. Combined with weather-adaptive scheduling, better contractor selection, and milestone-based early warning systems, construction organisations can dramatically reduce delay rates and protect project budgets.

---

*Report generated from actual dataset analysis. All statistics, model results, and insights are derived from the project_data.csv dataset.*
