-- Monthly cohort retention in plain SQL (SQLite dialect).
-- customers(customer_id, acq_month)  acq_month is 'YYYY-MM-01'
-- orders(order_id, customer_id, order_date, revenue)  order_date is 'YYYY-MM-DD ...'
-- Result: one row per cohort and month since acquisition with distinct active customers.

WITH activity AS (
    SELECT
        c.acq_month,
        (CAST(strftime('%Y', o.order_date) AS INTEGER) - CAST(strftime('%Y', c.acq_month) AS INTEGER)) * 12
          + (CAST(strftime('%m', o.order_date) AS INTEGER) - CAST(strftime('%m', c.acq_month) AS INTEGER))
          AS months_since_acquisition,
        o.customer_id,
        o.revenue
    FROM orders o
    JOIN customers c USING (customer_id)
),
sizes AS (
    SELECT acq_month, COUNT(*) AS cohort_size
    FROM customers
    GROUP BY acq_month
)
SELECT
    a.acq_month,
    a.months_since_acquisition,
    COUNT(DISTINCT a.customer_id) AS active_customers,
    SUM(a.revenue)                AS revenue,
    s.cohort_size,
    1.0 * COUNT(DISTINCT a.customer_id) / s.cohort_size AS retention
FROM activity a
JOIN sizes s USING (acq_month)
GROUP BY a.acq_month, a.months_since_acquisition
ORDER BY a.acq_month, a.months_since_acquisition;
