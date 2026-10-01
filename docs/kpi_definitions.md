# KPI definitions

| KPI | Definition | Grain |
|---|---|---|
| Customers | Distinct customers with at least one order up to the snapshot | Snapshot |
| Revenue | Sum of order revenue | Order, month, customer |
| Orders | Count of orders | |
| Average order value | `Revenue / Orders` | |
| Recency | Days from the last order to the snapshot | Customer |
| Frequency | Orders up to the snapshot | Customer |
| Monetary value | Revenue up to the snapshot | Customer |
| RFM score | Rank-based quintile (1 to 5) of recency (recent is 5), frequency and monetary value | Customer |
| Cohort | Customers grouped by month of first order | |
| Active in month | Customer placed at least one order in the calendar month | |
| Retention (month n) | `Active customers n months after acquisition / cohort size` | Cohort and age |
| Revenue per customer (month n) | `Cohort revenue in month n / cohort size` | Cohort and age |
| Churn (model target) | No order in the 90 days after the snapshot, among customers active in the 180 days before it | Customer |
| 6-month value | Revenue in the 182 days after the snapshot | Customer |
| 30-day repurchase rate | Share of a group ordering within 30 days of the snapshot | Group |
| Absolute lift | `Rate(treatment) - Rate(control)`, in percentage points | Experiment |
| Relative lift | `Absolute lift / Rate(control)` | Experiment |
| Power | Probability of detecting the planned effect with the available sample | Experiment |
| MDE | Smallest absolute lift detectable at 80% power with the available sample | Experiment |
| MAPE | Mean of `abs(actual - forecast) / actual` | Forecast |
