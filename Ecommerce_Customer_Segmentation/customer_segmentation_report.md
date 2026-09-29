# Customer Segmentation Report

## Objective
Identify distinct customer groups from ecommerce transaction data to enable targeted marketing and retention strategies.

## Dataset Overview
| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Age, Gender | Demographics |
| City | Location |
| MembershipType | Bronze / Silver / Gold |
| TotalSpend | Cumulative spend |
| ItemsPurchased | Total items bought |
| DaysSinceLastPurchase | Recency indicator |
| SatisfactionScore | 1–5 rating |

## Methodology

### 1. Data Cleaning
- Removed duplicates and null rows
- Encoded `Gender` (Male=0, Female=1) and `MembershipType` (Bronze=0, Silver=1, Gold=2)

### 2. Exploratory Data Analysis
- Distribution plots for spend, age, frequency, recency
- Correlation heatmap to identify feature relationships
- Boxplot of spend across membership tiers

### 3. RFM Segmentation
Customers scored on:
- **Recency** — Days since last purchase (lower = better)
- **Frequency** — Number of items purchased
- **Monetary** — Total spend

Combined RFM score mapped to 3 segments:

| Segment | RFM Score | Profile |
|---|---|---|
| High | 9–12 | Loyal, high-value customers |
| Mid | 6–8 | Occasional buyers with growth potential |
| Low | 3–5 | At-risk or inactive customers |

### 4. KNN Classification
- Features: Age, TotalSpend, ItemsPurchased, DaysSinceLastPurchase, AveragePurchaseValue, SatisfactionScore, MembershipType
- Best K selected via accuracy elbow method
- Model saved to `outputs/models/knn_model.pkl`

## Key Findings
- **High-value customers** tend to be Gold members with low recency and high spend
- **Low-value customers** show high recency (inactive) and low satisfaction scores
- MembershipType and TotalSpend are the strongest predictors of segment

## Recommendations
| Segment | Action |
|---|---|
| High | Loyalty rewards, early access offers |
| Mid | Re-engagement campaigns, upsell promotions |
| Low | Win-back discounts, satisfaction surveys |
