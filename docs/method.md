# Method

## Simulated world
Customers are acquired every month for 36 months (400 in the first month, growing by 5 per month). Each belongs to a hidden segment:

| Segment | Share | Orders per month while active | Monthly dropout | Average order value |
|---|---:|---:|---:|---:|
| Loyalist | 15% | 0.55 | 1.0% | 85 |
| Bargain hunter | 30% | 0.30 | 3.5% | 42 |
| Occasional | 35% | 0.10 | 2.5% | 64 |
| One and done | 20% | 0.03 | 10.0% | 50 |

Purchases are Poisson with a calendar-month multiplier (low in winter, 1.35 in November, 1.5 in December). Every customer's first order lands in the acquisition month. Order values are lognormal. Acquisition channel depends on segment.

## Point-in-time features (no leakage)
Two snapshots are used: **2024-06-30** for training and **2025-06-30** for testing. For a snapshot, features (recency, frequency, monetary value, tenure, last-90-day activity, average order value, orders per month, channel) use only orders on or before the snapshot. Outcomes use only orders after it. A test verifies that deleting every later order leaves the features unchanged.

## Segmentation
* **RFM:** recency, frequency and monetary value are scored 1 to 5 by rank so quintiles are balanced. Rules map scores to Champions, Loyal, Promising, Needs attention, At risk and Hibernating. Segments are validated by their *future* repurchase rates.
* **K-means:** on standardised log recency, frequency and monetary value; k from 3 to 7 chosen by silhouette on a sample; agreement with the hidden segments measured by adjusted Rand index.

## Cohort retention
Customers are grouped by acquisition month. A customer is active in a month if they ordered. Only ages that are fully observed are shown. The SQL version uses `strftime` month arithmetic and `COUNT(DISTINCT ...)`; a test requires identical counts, revenue and rates.

## Churn and value models
* **Population:** customers who bought in the last 180 days, the group a retention team can still influence.
* **Churn:** no order in the next 90 days. Logistic regression (on log-transformed, standardised features) and gradient boosting. Boosting hyper-parameters (depth 2 or 3, 50, 100 or 200 trees) are chosen on a validation split of the *training* snapshot only.
* **Value:** gradient boosting regressor for revenue in the next 182 days, compared with a run-rate baseline (spend per tenure month times six).
* **Evaluation:** train on 2024-06-30, evaluate on 2025-06-30 (out of time). Metrics: ROC AUC, average precision, top-decile churn rate and lift, MAE, RMSE, Spearman correlation and top-decile revenue capture.

## Forecast
Monthly revenue; Holt-Winters with damped additive trend and multiplicative seasonality (period 12), last six months held out. Baselines: last month repeated, and same month last year scaled by the last 12 months' growth. Error is MAPE.

## Win-back experiment
* **Population:** lapsed customers (recency score 1 or 2) at 2025-06-30.
* **Assignment:** random 50/50, checked with a sample-ratio-mismatch chi-square test.
* **Outcome:** observed 30-day repurchase from the simulated orders. A treatment effect is then layered on: a share of treated non-purchasers are flipped to purchasers so that the expected absolute lift is +2.0 pp, with extra order value drawn from a lognormal. The effect, assignment and extra revenue are synthetic.
* **Analysis:** two-proportion z test (pooled standard error for the p-value, unpooled for the interval), Bayesian beta-binomial read-out with uniform priors, Welch test for revenue per customer, CUPED using lifetime spend as the pre-experiment covariate, subgroup lifts by channel with Benjamini-Hochberg correction, and a likelihood-ratio **interaction test** for whether the effect differs by channel.
* **Planning:** sample size per arm and power from the closed-form normal approximation, and the smallest detectable effect with the available sample.

## Validating the statistics
`validate_methods` runs thousands of simulated experiments: A/A tests should reject 5% of the time and 95% intervals should cover the truth 95% of the time; A/B tests should reject at the rate the power formula predicts.
