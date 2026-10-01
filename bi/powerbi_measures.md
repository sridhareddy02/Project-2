# Power BI model and measures

Import the CSVs in `bi/data/` (Get data > Text/CSV). Set `acq_month` and `month` to Date.

## Relationships
`dim_customer_segments[customer_id]` is the customer dimension. `fact_cohort_retention` is aggregated by `acq_month` and
`months_since_acquisition`; relate `fact_cohort_retention[acq_month]` to `dim_customer_segments[acq_month]` (many to many, single direction) only if
you need to slice cohorts by segment, otherwise leave the table stand-alone.

## Measures

```DAX
Customers = DISTINCTCOUNT ( dim_customer_segments[customer_id] )

Revenue to Date = SUM ( dim_customer_segments[monetary] )

Avg Orders per Customer = DIVIDE ( SUM ( dim_customer_segments[frequency] ), [Customers] )

Customer Share = DIVIDE ( [Customers], CALCULATE ( [Customers], ALL ( dim_customer_segments[rfm_segment] ) ) )

Revenue Share = DIVIDE ( [Revenue to Date], CALCULATE ( [Revenue to Date], ALL ( dim_customer_segments[rfm_segment] ) ) )

Cohort Size = MAX ( fact_cohort_retention[cohort_size] )

Retention % =
DIVIDE ( SUM ( fact_cohort_retention[active_customers] ), SUM ( fact_cohort_retention[cohort_size] ) )

Revenue per Customer =
DIVIDE ( SUM ( fact_cohort_retention[revenue] ), SUM ( fact_cohort_retention[cohort_size] ) )

Monthly Revenue = SUM ( fact_monthly_revenue[revenue] )

Revenue MoM % =
VAR prev = CALCULATE ( [Monthly Revenue], DATEADD ( fact_monthly_revenue[month], -1, MONTH ) )
RETURN DIVIDE ( [Monthly Revenue] - prev, prev )
```

Note: `Retention %` sums `cohort_size` across rows for the same cohort, so put `months_since_acquisition` on the columns of a matrix and
`acq_month` on the rows and it is correct per cell. For an average curve across cohorts use the measure with only `months_since_acquisition` on the axis.

## Suggested pages
1. **Segments:** bars for `Customer Share` and `Revenue Share` by `rfm_segment`, plus a table from `fact_segment_summary` with the 90-day repurchase rate.
2. **Retention:** matrix of `Retention %` (rows `acq_month`, columns `months_since_acquisition`) with a colour scale.
3. **Revenue:** line chart of `Monthly Revenue` and `Revenue MoM %`.
4. **Value model:** line chart of `avg_predicted` and `avg_actual` by `decile` from `fact_value_deciles`.
