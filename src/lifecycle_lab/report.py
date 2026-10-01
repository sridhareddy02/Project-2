"""Self-contained HTML report."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

CSS = """
body{font:16px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1b1f2a;max-width:940px;margin:40px auto;padding:0 20px}
h1{font-size:30px;margin-bottom:4px} h2{margin-top:40px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}
.sub{color:#6b7280;margin-top:0} .note{background:#fef9c3;border:1px solid #fde68a;padding:10px 14px;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0} th,td{padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:right}
th:first-child,td:first-child{text-align:left} th{background:#f3f4f6} img{max-width:100%;border:1px solid #e5e7eb;border-radius:8px}
"""


def _img(path: Path) -> str:
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, float_format=lambda v: f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}", na_rep="n/a")


def build_report(out: Path, meta: dict, findings: list[str], tables: dict[str, pd.DataFrame]) -> Path:
    img = {k: out / f"{k}.png" for k in ("segments", "cohorts", "churn", "value", "forecast", "experiment", "validation")}
    parts = [
        f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Lifecycle lab report</title><style>{CSS}</style><body>",
        "<h1>Customer lifecycle and retention lab</h1>",
        f"<p class='sub'>Simulated data, seed {meta['seed']}: {meta['customers']:,} customers, {meta['orders']:,} orders, "
        f"{meta['months']} months.</p>",
        "<p class='note'><b>All numbers on this page come from a seeded simulation.</b> Nothing here is real customer data, "
        "and the win-back experiment's treatment effect is synthetic.</p>",
        "<h2>Key findings</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in findings) + "</ul>",
        "<h2>1. Segmentation</h2>", _img(img["segments"]), _table(tables["segments"]),
        "<h3>K-means clusters</h3>", _table(tables["clusters"]),
        "<h2>2. Cohort retention</h2>", _img(img["cohorts"]),
        "<h2>3. Churn risk</h2>", _img(img["churn"]), _table(tables["churn"]),
        "<h2>4. Customer value</h2>", _img(img["value"]), _table(tables["value"]),
        "<h2>5. Revenue forecast</h2>", _img(img["forecast"]), _table(tables["forecast"]),
        "<h2>6. Win-back experiment</h2>", _img(img["experiment"]), _table(tables["experiment"]),
        "<h3>Does the testing method itself work?</h3>", _img(img["validation"]),
        "<h2>Limitations</h2><ul><li>Simulated customers follow simple purchase processes; real behaviour is messier.</li>"
        "<li>The experiment's treatment effect is injected by the simulator, so the lift is known in advance.</li>"
        "<li>Forecast accuracy reflects a smooth synthetic series.</li></ul></body></html>",
    ]
    path = out / "report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
