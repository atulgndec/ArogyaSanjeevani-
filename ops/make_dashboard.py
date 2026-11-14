"""Generates the Grafana dashboard JSON. Run: python ops/make_dashboard.py"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}      # matches uid in grafana/provisioning/datasources

# (title, query, panel type, legend). Label breakdowns are bar gauges (one labelled bar per value). The legend names only the labels the query groups by,
# so each series gets one clear name. Single-value stat panels use no legend at all.
PANELS = [
    ("Claims by outcome", "sum by (outcome) (arogya_claims_total)", "bargauge", "{{outcome}}"),
    ("Approvals waiting", "arogya_approvals_pending", "stat", None),
    ("Rupees settled", "arogya_settled_rupees_total", "stat", None),
    ("Rupees deducted by reason", "sum by (reason) (arogya_deducted_rupees_total)", "bargauge", "{{reason}}"),
    ("Fraud flags by type", "sum by (type) (arogya_fraud_flags_total)", "bargauge", "{{type}}"),
    ("Node time p95 (s)", "histogram_quantile(0.95, sum by (le, node) (rate(arogya_node_seconds_bucket[5m])))", "timeseries", "{{node}}"),
    ("Model calls by mode", "sum by (mode) (arogya_llm_calls_total)", "piechart", "{{mode}}"),
    ("Adjudicator decisions", "sum by (action, level) (arogya_decisions_total)", "bargauge", "{{action}} / {{level}}"),
    ("Critic send-backs", "arogya_critic_revisions_total", "stat", None),
    ("Lessons by status", "sum by (status) (arogya_lessons)", "bargauge", "{{status}}"),
]


def panel(i, title, expr, kind, legend):
    target = {"expr": expr, "refId": "A"}
    if legend:
        target["legendFormat"] = legend
    p = {"id": i + 1, "title": title, "type": kind, "datasource": DS,
         "gridPos": {"x": (i % 3) * 8, "y": (i // 3) * 8, "w": 8, "h": 8}, "targets": [target]}
    if kind == "stat":
        p["options"] = {"textMode": "value"}          # show just the number; the panel title says what it is
    if kind in ("bargauge", "piechart"):               # one value per label: ask Prometheus for the latest sample only,
        target["instant"], target["range"] = True, False   # not a time series (a bar chart over time was one bar per scrape)
    if kind == "bargauge":                            # one labelled horizontal bar per series; the legend text is the bar's name
        p["options"] = {"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True,
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}
    return p


panels = [panel(i, *row) for i, row in enumerate(PANELS)]
Path(__file__).parent.joinpath("grafana/dashboards/arogya.json").write_text(
    json.dumps({"title": "Arogya claims", "uid": "arogya", "schemaVersion": 39, "refresh": "10s", "panels": panels}, indent=1))


# """Generates the Grafana dashboard JSON. Run: python ops/make_dashboard.py"""
# import json
# from pathlib import Path

# PANELS = [
#     ("Claims by outcome", "sum by (outcome) (arogya_claims_total)", "barchart"),
#     ("Approvals waiting", "arogya_approvals_pending", "stat"),
#     ("Rupees settled", "arogya_settled_rupees_total", "stat"),
#     ("Rupees deducted by reason", "sum by (reason) (arogya_deducted_rupees_total)", "barchart"),
#     ("Fraud flags by type", "sum by (type) (arogya_fraud_flags_total)", "barchart"),
#     ("Node time p95 (s)", "histogram_quantile(0.95, sum by (le, node) (rate(arogya_node_seconds_bucket[5m])))", "timeseries"),
#     ("Model calls by mode", "sum by (mode) (arogya_llm_calls_total)", "piechart"),
#     ("Adjudicator decisions", "sum by (action, level) (arogya_decisions_total)", "barchart"),
#     ("Critic send-backs", "arogya_critic_revisions_total", "stat"),
#     ("Lessons by status", "arogya_lessons", "barchart"),
# ]
# panels = [{"id": i + 1, "title": t, "type": ty, "datasource": "Prometheus", "gridPos": {"x": (i % 3) * 8, "y": (i // 3) * 8, "w": 8, "h": 8},
#            "targets": [{"expr": e, "legendFormat": "{{outcome}}{{reason}}{{type}}{{node}}{{mode}}{{action}}{{status}}", "refId": "A"}]} for i, (t, e, ty) in enumerate(PANELS)]
# Path(__file__).parent.joinpath("grafana/dashboards/arogya.json").write_text(json.dumps({"title": "Arogya claims", "uid": "arogya", "schemaVersion": 39, "refresh": "10s", "panels": panels}, indent=1))
