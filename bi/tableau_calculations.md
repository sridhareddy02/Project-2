# Tableau setup

Connect to the CSVs in `bi/data/`.

```
// Retention % (cohort and age on the view)
SUM([active_customers]) / SUM([cohort_size])

// Revenue per customer
SUM([revenue]) / SUM([cohort_size])

// Share of customers (table calculation, compute using Segment)
SUM([customers]) / TOTAL(SUM([customers]))

// Share of revenue (table calculation, compute using Segment)
SUM([revenue_to_date]) / TOTAL(SUM([revenue_to_date]))

// Month over month revenue change
(SUM([revenue]) - LOOKUP(SUM([revenue]), -1)) / LOOKUP(SUM([revenue]), -1)
```

Suggested views: a highlight table of retention (cohort by months since acquisition), a bar chart of customer share against revenue share by segment, and a line chart of monthly revenue.
