import io
import json
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import ollama


# ---------- JSON-safe conversion helper ----------

def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(i) for i in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------- AI helper context builders ----------

def build_compact_context(gc_df=None, app_df=None, correlations=None, max_rows: int = 200) -> str:
    parts = []

    if gc_df is not None and isinstance(gc_df, pd.DataFrame) and not gc_df.empty:
        sample_gc = gc_df.head(max_rows).to_dict(orient="records")
        parts.append(f"GC_LOG_EVENTS_JSON = {convert(sample_gc)}")

    if app_df is not None and isinstance(app_df, pd.DataFrame) and not app_df.empty:
        sample_app = app_df.head(max_rows).to_dict(orient="records")
        parts.append(f"APP_LOG_EVENTS_JSON = {convert(sample_app)}")

    if correlations is not None and isinstance(correlations, pd.DataFrame) and not correlations.empty:
        sample_corr = correlations.head(max_rows).to_dict(orient="records")
        parts.append(f"CORRELATIONS_JSON = {convert(sample_corr)}")

    return "\n\n".join(parts) if parts else "No parsed data available."


def build_pattern_signals(gc_df=None, app_df=None) -> dict:
    signals = {}

    if gc_df is not None and isinstance(gc_df, pd.DataFrame) and not gc_df.empty:
        if "pause_ms" in gc_df.columns:
            signals["gc_avg_pause_ms"] = float(gc_df["pause_ms"].mean())
            signals["gc_max_pause_ms"] = float(gc_df["pause_ms"].max())
        signals["gc_event_count"] = int(len(gc_df))
        if "heap_after_k" in gc_df.columns:
            signals["heap_after_k_mean"] = float(gc_df["heap_after_k"].tail(50).mean())

    if app_df is not None and isinstance(app_df, pd.DataFrame) and not app_df.empty:
        if "level" in app_df.columns:
            signals["app_error_count"] = int((app_df["level"] == "ERROR").sum())
            signals["app_warn_count"] = int((app_df["level"] == "WARN").sum())
        signals["app_event_count"] = int(len(app_df))

    return signals


# ---------- GC log parser ----------

def parse_gc_log(file) -> pd.DataFrame:
    text = file.read().decode("utf-8", errors="ignore")
    lines = [l for l in text.splitlines() if l.strip()]
    records = []

    for line in lines:
        ts_match = re.match(
            r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[\+\-]\d+)",
            line
        )
        ts = ts_match.group(1) if ts_match else None

        dur_match = re.search(r"(\d+\.\d+)\s*secs", line)
        duration_s = float(dur_match.group(1)) if dur_match else 0.0

        mem_match = re.search(r"(\d+)K->(\d+)K\((\d+)K\)", line)
        if mem_match:
            before_k = int(mem_match.group(1))
            after_k = int(mem_match.group(2))
            total_k = int(mem_match.group(3))
        else:
            before_k = after_k = total_k = None

        if "Full GC" in line:
            gc_type = "Full GC"
        elif "GC" in line:
            gc_type = "GC"
        else:
            gc_type = "Unknown"

        records.append({
            "raw": line,
            "timestamp": ts,
            "event": gc_type,
            "duration_s": duration_s,
            "heap_before_k": before_k,
            "heap_after_k": after_k,
            "heap_total_k": total_k,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["gc_time"] = (
            pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
              .dt.tz_convert(None)
        )
        df["pause_ms"] = df["duration_s"] * 1000
        df["gc_category"] = df["event"].apply(
            lambda x: "Full GC" if "Full" in str(x) else
                      "Young/Mixed GC" if "GC" in str(x) else
                      "Other"
        )
        df["severity"] = df["pause_ms"].apply(
            lambda x: "High" if x >= 200 else "Medium" if x >= 50 else "Low"
        )
    return df


# ---------- Application log parser ----------

def parse_app_log(file) -> pd.DataFrame:
    text = file.read().decode("utf-8", errors="ignore")
    lines = [l for l in text.splitlines() if l.strip()]
    records = []

    for line in lines:
        match = re.match(
            r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(INFO|WARN|ERROR)\s+(.*)$",
            line
        )
        if match:
            ts, level, msg = match.groups()
        else:
            ts, level, msg = None, "INFO", line

        msg_lower = msg.lower()
        if any(k in msg_lower for k in ["db", "database", "query"]):
            category = "Database"
        elif any(k in msg_lower for k in ["timeout", "unreachable", "network"]):
            category = "Network"
        elif any(k in msg_lower for k in ["memory", "heap", "outofmemory"]):
            category = "Memory"
        else:
            category = "General"

        records.append({
            "raw": line,
            "timestamp": ts,
            "level": level,
            "message": msg,
            "category": category,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["app_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# ---------- Correlation + severity scoring ----------

def score_correlation(gc_pause_ms, app_level, time_diff_ms):
    level_weight = {"ERROR": 3, "WARN": 2, "INFO": 1}
    w_level = level_weight.get(str(app_level), 1)

    pause_score = 0
    if gc_pause_ms is not None:
        if gc_pause_ms >= 200:
            pause_score = 3
        elif gc_pause_ms >= 50:
            pause_score = 2
        else:
            pause_score = 1

    time_score = 0
    if time_diff_ms is not None:
        adiff = abs(time_diff_ms)
        if adiff <= 2000:
            time_score = 3
        elif adiff <= 5000:
            time_score = 2
        else:
            time_score = 1

    total = w_level + pause_score + time_score
    if total >= 8:
        severity = "High"
    elif total >= 5:
        severity = "Medium"
    else:
        severity = "Low"

    return total, severity


def correlate(gc_df: pd.DataFrame, app_df: pd.DataFrame, window_size: int, spike_factor: float) -> pd.DataFrame:
    if gc_df is None or app_df is None or gc_df.empty or app_df.empty:
        return pd.DataFrame()

    if "gc_time" not in gc_df.columns:
        gc_df["gc_time"] = (
            pd.to_datetime(gc_df["timestamp"], errors="coerce", utc=True)
              .dt.tz_convert(None)
        )
    if "app_time" not in app_df.columns:
        app_df["app_time"] = pd.to_datetime(app_df["timestamp"], errors="coerce")

    gc_valid = gc_df.dropna(subset=["gc_time"])
    app_valid = app_df.dropna(subset=["app_time"])

    if gc_valid.empty or app_valid.empty:
        return pd.DataFrame()

    window_ms = window_size * 1000
    rows = []

    for _, gc in gc_valid.iterrows():
        diffs = (app_valid["app_time"] - gc["gc_time"]).dt.total_seconds() * 1000
        matches = app_valid[diffs.abs() <= window_ms].copy()
        if matches.empty:
            continue

        matches["time_diff_ms"] = diffs[matches.index]

        for _, app in matches.iterrows():
            gc_pause_ms = gc.get("pause_ms", gc.get("duration_s", 0.0) * 1000)
            time_diff_ms = app.get("time_diff_ms")
            score, severity = score_correlation(gc_pause_ms, app.get("level"), time_diff_ms)

            rows.append({
                "gc_timestamp": gc.get("timestamp"),
                "gc_event": gc.get("event"),
                "gc_pause_ms": gc_pause_ms,
                "heap_before_k": gc.get("heap_before_k"),
                "heap_after_k": gc.get("heap_after_k"),
                "heap_total_k": gc.get("heap_total_k"),
                "app_timestamp": app.get("timestamp"),
                "app_level": app.get("level"),
                "app_category": app.get("category"),
                "app_message": app.get("message"),
                "time_diff_ms": time_diff_ms,
                "correlation_score": score,
                "correlation_severity": severity,
                "correlation_type": "Temporal Match",
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------- Grouped correlation view ----------

def build_grouped_correlations(corr_df: pd.DataFrame):
    if corr_df is None or corr_df.empty:
        return []

    grouped = []
    for (gc_ts, gc_event), group in corr_df.groupby(["gc_timestamp", "gc_event"]):
        group_sorted = group.sort_values("time_diff_ms")
        app_events = []
        for _, row in group_sorted.iterrows():
            app_events.append({
                "app_timestamp": row["app_timestamp"],
                "app_level": row["app_level"],
                "app_category": row["app_category"],
                "app_message": row["app_message"],
                "time_diff_ms": row["time_diff_ms"],
                "correlation_severity": row["correlation_severity"],
                "correlation_score": row["correlation_score"],
            })

        grouped.append({
            "gc_timestamp": gc_ts,
            "gc_event": gc_event,
            "gc_pause_ms": group_sorted["gc_pause_ms"].iloc[0],
            "heap_before_k": group_sorted["heap_before_k"].iloc[0],
            "heap_after_k": group_sorted["heap_after_k"].iloc[0],
            "heap_total_k": group_sorted["heap_total_k"].iloc[0],
            "severity_max": group_sorted["correlation_severity"].max(),
            "app_events": app_events,
        })

    return grouped


# ---------- AI insight generation (with modes) ----------

def generate_ai_insight(data, model: str, mode: str):
    if isinstance(data, pd.DataFrame):
        safe = convert(data.to_dict(orient="records"))
    else:
        safe = convert(data)

    if mode == "full":
        context = (
            "You are analyzing correlated JVM GC logs and application logs. "
            "Focus on how GC pauses impact application behavior, errors, latency, and resource usage."
        )
    elif mode == "gc_only":
        context = (
            "You are analyzing ONLY JVM GC logs. "
            "Focus on memory pressure, pause times, GC frequency, heap behavior, and probable application impact."
        )
    elif mode == "app_only":
        context = (
            "You are analyzing ONLY application logs (no GC data). "
            "Focus on errors, warnings, latency spikes, memory errors, and service health patterns."
        )
    else:
        context = "You are analyzing JVM-related logs."

    prompt = f"""
{context}

You MUST respond with ONLY valid JSON.
No markdown. No backticks. No commentary.

JSON schema:
{{
  "root_cause": "",
  "impact": "",
  "evidence": [
    "First key evidence point",
    "Second key evidence point"
  ],
  "recommendations": [
    "First concrete recommendation",
    "Second concrete recommendation"
  ],
  "confidence": 0.0,
  "confidence_explanation": "",
  "next_steps": [
    "Immediate next step",
    "Follow-up step"
  ]
}}

Data:
{json.dumps(safe, indent=2)}
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"]

    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw or "")
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "root_cause": "Model did not return valid JSON.",
        "impact": (raw or "")[:500],
        "evidence": [],
        "recommendations": ["Try a smaller model or adjust the prompt."],
        "confidence": 0.0,
        "confidence_explanation": "Fallback because response was not valid JSON.",
        "next_steps": [],
    }


# ---------- AI-driven high-level analysis features ----------

def ai_root_cause_analysis(gc_df=None, app_df=None, correlations=None, model: str = "gpt-oss:120b-cloud") -> str:
    context = build_compact_context(gc_df, app_df, correlations)

    system_prompt = (
        "You are an expert JVM performance engineer. "
        "Given GC logs, application logs, and correlations, "
        "identify the single most likely root cause of performance issues. "
        "Be concise, structured, and practical."
    )

    user_prompt = f"""
Context:
{context}

Task:
1. Identify the most likely root cause.
2. Explain the evidence in 3–5 bullet points.
3. Suggest 3 concrete actions to try first.
4. Provide a confidence score between 0 and 100.

Format exactly as:

Root Cause:
<one sentence>

Evidence:
- <bullet>
- <bullet>

Actions:
1. <action>
2. <action>
3. <action>

Confidence: <number>/100
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": system_prompt + "\n\n" + user_prompt}],
    )
    return response["message"]["content"].strip()


def ai_pattern_detection(gc_df=None, app_df=None, model: str = "gpt-oss:120b-cloud") -> str:
    context = build_compact_context(gc_df, app_df)
    signals = build_pattern_signals(gc_df, app_df)

    system_prompt = (
        "You are a JVM performance expert. "
        "Given log context and summary metrics, detect meaningful performance patterns."
    )

    user_prompt = f"""
Context:
{context}

Signals:
{json.dumps(convert(signals), indent=2)}

Task:
Identify performance patterns.

For each pattern, provide:
- Name (e.g., "GC Thrashing", "Memory Leak Suspected", "High Error Rate")
- Severity (Low/Medium/High/Critical)
- Evidence (1–2 bullets)
- Impact (short sentence)
- Recommended next step

Format:

Pattern:
Name: ...
Severity: ...
Evidence:
- ...
Impact: ...
Next Step: ...
---
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": system_prompt + "\n\n" + user_prompt}],
    )
    return response["message"]["content"].strip()


def ai_timeline_narration(gc_df=None, app_df=None, correlations=None, model: str = "gpt-oss:120b-cloud") -> str:
    context = build_compact_context(gc_df, app_df, correlations)

    system_prompt = (
        "You are narrating JVM and application behavior over time. "
        "Create a concise chronological story of key events."
    )

    user_prompt = f"""
Context:
{context}

Task:
1. Describe the key timeline in 5–10 chronological points.
2. Focus on spikes, anomalies, and interactions between GC and application behavior.
3. Use timestamps or relative phrases ("around 14:32") when possible.
4. End with 2–3 bullets summarizing what really happened.

Format:

Timeline:
1. ...
2. ...
...

Summary:
- ...
- ...
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": system_prompt + "\n\n" + user_prompt}],
    )
    return response["message"]["content"].strip()


def ai_log_summaries(gc_df=None, app_df=None, model: str = "gpt-oss:120b-cloud") -> str:
    context = build_compact_context(gc_df, app_df)

    system_prompt = (
        "You summarize JVM GC and application logs for performance engineers. "
        "Be sharp, structured, and avoid fluff."
    )

    user_prompt = f"""
Context:
{context}

Task:
Provide three sections:

GC Summary:
- ...

Application Summary:
- ...

Combined Behavior:
- ...

Each section: 3–5 bullets, focusing on performance, stability, and anomalies.
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": system_prompt + "\n\n" + user_prompt}],
    )
    return response["message"]["content"].strip()


# ---------- Text report ----------

def build_text_report(gc_df, app_df, correlations, insight, mode: str) -> str:
    if insight is None:
        insight = {
            "root_cause": "Insight not generated.",
            "impact": "",
            "evidence": [],
            "recommendations": [],
            "confidence": 0.0,
            "confidence_explanation": "",
            "next_steps": [],
        }

    lines = []
    lines.append("JVM AI Observability Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. Session Overview")
    lines.append(f"- Mode: {mode}")
    lines.append(f"- GC events parsed: {len(gc_df) if gc_df is not None else 0}")
    lines.append(f"- App log entries parsed: {len(app_df) if app_df is not None else 0}")
    lines.append(f"- Correlated events: {len(correlations) if isinstance(correlations, pd.DataFrame) else 0}")
    lines.append("")

    if isinstance(correlations, pd.DataFrame):
        corr_json = correlations.to_dict(orient="records")
    else:
        corr_json = convert(correlations) if correlations is not None else []

    lines.append("2. Correlation Summary (JSON)")
    lines.append(json.dumps(corr_json, indent=2))
    lines.append("")

    lines.append("3. AI Insight")
    lines.append(f"Root cause: {insight.get('root_cause', '')}")
    lines.append(f"Impact: {insight.get('impact', '')}")
    lines.append(f"Confidence: {insight.get('confidence', '')}")
    lines.append(f"Confidence explanation: {insight.get('confidence_explanation', '')}")
    lines.append("")
    lines.append("Evidence:")
    for e in insight.get("evidence", []):
        lines.append(f"- {e}")
    lines.append("")
    lines.append("Recommendations:")
    for r in insight.get("recommendations", []):
        lines.append(f"- {r}")
    lines.append("")
    lines.append("Next steps:")
    for s in insight.get("next_steps", []):
        lines.append(f"- {s}")
    lines.append("")

    return "\n".join(lines)


# ---------- HTML report builder ----------

def build_html_report(gc_df, app_df, correlations, insight, mode: str) -> str:
    if insight is None:
        insight = {
            "root_cause": "Insight not generated.",
            "impact": "",
            "evidence": [],
            "recommendations": [],
            "confidence": 0.0,
            "confidence_explanation": "",
            "next_steps": [],
        }

    gc_count = len(gc_df) if gc_df is not None else 0
    app_count = len(app_df) if app_df is not None else 0
    corr_count = len(correlations) if isinstance(correlations, pd.DataFrame) else 0

    gc_table_html = gc_df.to_html(index=False, classes="table table-sm table-striped") if gc_df is not None and not gc_df.empty else ""
    app_table_html = app_df.to_html(index=False, classes="table table-sm table-striped") if app_df is not None and not app_df.empty else ""
    corr_table_html = correlations.to_html(index=False, classes="table table-sm table-striped") if isinstance(correlations, pd.DataFrame) and not correlations.empty else ""

    top_gc = (
        gc_df.sort_values("pause_ms", ascending=False)
        .head(10)
        .to_html(index=False, classes="table table-sm table-striped")
        if gc_df is not None and not gc_df.empty
        else ""
    )

    top_errors = (
        app_df[app_df["level"] == "ERROR"]
        .head(10)
        .to_html(index=False, classes="table table-sm table-striped")
        if app_df is not None and not app_df.empty
        else ""
    )

    top_corr = (
        correlations.sort_values("correlation_score", ascending=False)
        .head(10)
        .to_html(index=False, classes="table table-sm table-striped")
        if isinstance(correlations, pd.DataFrame) and not correlations.empty
        else ""
    )

    evidence_items = "".join(f"<li>{e}</li>" for e in insight.get("evidence", [])[:10])
    rec_items = "".join(f"<li>{r}</li>" for r in insight.get("recommendations", [])[:10])
    next_items = "".join(f"<li>{s}</li>" for s in insight.get("next_steps", [])[:10])

    gc_tab_button = ""
    gc_tab_pane = ""
    if gc_df is not None and not gc_df.empty:
        gc_tab_button = """
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="gc-tab" data-bs-toggle="tab" data-bs-target="#gc" type="button" role="tab">GC Details</button>
    </li>
"""
        gc_tab_pane = f"""
    <div class="tab-pane fade" id="gc" role="tabpanel" aria-labelledby="gc-tab">
      <h3 class="mt-3">GC Details (full)</h3>
      {gc_table_html or "<p><em>No GC data available.</em></p>"}
    </div>
"""

    app_tab_button = ""
    app_tab_pane = ""
    if app_df is not None and not app_df.empty:
        app_tab_button = """
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="app-tab" data-bs-toggle="tab" data-bs-target="#app" type="button" role="tab">App Log Details</button>
    </li>
"""
        app_tab_pane = f"""
    <div class="tab-pane fade" id="app" role="tabpanel" aria-labelledby="app-tab">
      <h3 class="mt-3">Application Log Details (full)</h3>
      {app_table_html or "<p><em>No App log data available.</em></p>"}
    </div>
"""

    corr_tab_button = ""
    corr_tab_pane = ""
    if isinstance(correlations, pd.DataFrame) and not correlations.empty:
        corr_tab_button = """
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="corr-tab" data-bs-toggle="tab" data-bs-target="#corr" type="button" role="tab">Correlations</button>
    </li>
"""
        corr_tab_pane = f"""
    <div class="tab-pane fade" id="corr" role="tabpanel" aria-labelledby="corr-tab">
      <h3 class="mt-3">Correlations (full)</h3>
      {corr_table_html or "<p><em>No correlation data available.</em></p>"}
    </div>
"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>JVM AI Observability Report</title>
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
      rel="stylesheet"
    >
    <style>
      body {{ padding: 20px; background: #020617; color: #E5E7EB; }}
      .tab-content {{ margin-top: 20px; }}
      pre {{ white-space: pre-wrap; background:#020617; color:#E5E7EB; padding:12px; border-radius:8px; }}
      .nav-tabs .nav-link.active {{
          background-color: #0F172A;
          color: #67E8F9;
          border-color: #0EA5E9;
      }}
      .nav-tabs .nav-link {{
          color: #E5E7EB;
      }}
    </style>
</head>
<body>
<div class="container-fluid">
  <h1 class="mb-3">JVM AI Observability Report</h1>
  <p class="text-muted">Generated at {datetime.now().isoformat()} | Mode: {mode}</p>

  <ul class="nav nav-tabs" id="reportTabs" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link active" id="summary-tab" data-bs-toggle="tab" data-bs-target="#summary" type="button" role="tab">Summary & Highlights</button>
    </li>
    {gc_tab_button}
    {app_tab_button}
    {corr_tab_button}
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="ai-tab" data-bs-toggle="tab" data-bs-target="#ai" type="button" role="tab">AI Insight</button>
    </li>
  </ul>

  <div class="tab-content">

    <div class="tab-pane fade show active" id="summary" role="tabpanel" aria-labelledby="summary-tab">
      <h3 class="mt-3">Summary</h3>
      <ul>
        <li><strong>Mode:</strong> {mode}</li>
        <li><strong>GC events parsed:</strong> {gc_count}</li>
        <li><strong>App log entries parsed:</strong> {app_count}</li>
        <li><strong>Correlated events:</strong> {corr_count}</li>
        <li><strong>Root cause:</strong> {insight.get("root_cause", "")}</li>
        <li><strong>Impact:</strong> {insight.get("impact", "")}</li>
        <li><strong>Confidence:</strong> {insight.get("confidence", "")}</li>
      </ul>

      <h4>Key Evidence</h4>
      <ul>
        {evidence_items}
      </ul>

      <h4>Top Recommendations</h4>
      <ul>
        {rec_items}
      </ul>

      <h4>Next Steps</h4>
      <ul>
        {next_items}
      </ul>

      <hr/>
      <h3>Top 10 GC Pauses</h3>
      {top_gc or "<p><em>No GC data available.</em></p>"}

      <h3 class="mt-4">Top 10 ERROR Logs</h3>
      {top_errors or "<p><em>No ERROR logs available.</em></p>"}

      <h3 class="mt-4">Top 10 Correlations</h3>
      {top_corr or "<p><em>No correlations available.</em></p>"}
    </div>

    {gc_tab_pane}
    {app_tab_pane}
    {corr_tab_pane}

    <div class="tab-pane fade" id="ai" role="tabpanel" aria-labelledby="ai-tab">
      <h3 class="mt-3">AI Insight (raw JSON)</h3>
      <pre>{json.dumps(convert(insight), indent=2)}</pre>
    </div>

  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    return html


# ---------- ZIP builder ----------

def build_full_session_zip(gc_df, app_df, correlations, insight, text_report: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if gc_df is not None and not gc_df.empty:
            zf.writestr("gc_parsed.csv", gc_df.to_csv(index=False))
        if app_df is not None and not app_df.empty:
            zf.writestr("app_parsed.csv", app_df.to_csv(index=False))

        if isinstance(correlations, pd.DataFrame) and not correlations.empty:
            corr_json = correlations.to_dict(orient="records")
        elif correlations is not None:
            corr_json = convert(correlations)
        else:
            corr_json = []

        zf.writestr("correlations.json", json.dumps(corr_json, indent=2))
        zf.writestr("ai_insight.json", json.dumps(convert(insight), indent=2))
        zf.writestr("ai_insight_report.txt", text_report)

    buffer.seek(0)
    return buffer.getvalue()


# ---------- Plotly visualizations ----------

def plot_gc_timeline(gc_df: pd.DataFrame):
    if gc_df is None or gc_df.empty:
        return None
    fig = px.scatter(
        gc_df,
        x="gc_time",
        y="pause_ms",
        color="gc_category",
        size="pause_ms",
        hover_data=["event", "heap_before_k", "heap_after_k", "heap_total_k"],
        title="GC Timeline (pause duration over time)",
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Pause (ms)",
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font_color="#E5E7EB",
        legend=dict(bgcolor="#020617")
    )
    return fig


def plot_correlation_timeline(corr_df: pd.DataFrame):
    if corr_df is None or corr_df.empty:
        return None
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(corr_df["gc_timestamp"]),
        y=[0] * len(corr_df),
        mode="markers",
        marker=dict(size=10, color="#F97316"),
        name="GC Event",
        text=[
            f"{row['gc_event']} ({row['gc_pause_ms']} ms)"
            for _, row in corr_df.iterrows()
        ],
        hoverinfo="text+x",
    ))

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(corr_df["app_timestamp"]),
        y=[1] * len(corr_df),
        mode="markers",
        marker=dict(size=9, color="#38BDF8"),
        name="App Event",
        text=[
            f"{row['app_level']} - {row['app_message']}"
            for _, row in corr_df.iterrows()
        ],
        hoverinfo="text+x",
    ))

    fig.update_layout(
        title="Correlation Timeline (GC vs App Events)",
        xaxis_title="Time",
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["GC", "App"]
        ),
        showlegend=True,
        height=400,
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font_color="#E5E7EB",
        legend=dict(bgcolor="#020617")
    )
    return fig


def plot_gc_pause_heatmap(gc_df: pd.DataFrame):
    if gc_df is None or gc_df.empty:
        return None
    df = gc_df.copy()
    df["minute_bucket"] = df["gc_time"].dt.floor("min")
    pivot = df.pivot_table(
        index="minute_bucket",
        columns="gc_category",
        values="pause_ms",
        aggfunc="sum",
        fill_value=0,
    )
    if pivot.empty:
        return None
    fig = px.imshow(
        pivot.T,
        labels=dict(x="Time (minute)", y="GC Category", color="Total Pause (ms)"),
        aspect="auto",
        title="GC Pause Heatmap (by category and minute)",
        color_continuous_scale="Turbo",
    )
    fig.update_layout(
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font_color="#E5E7EB",
    )
    return fig


# ---------- Streamlit UI (AI theme + neat tabs) ----------

st.set_page_config(page_title="Logs Heap AI Analyzer", layout="wide")

st.markdown("""
<style>
.main .block-container {
    max-width: 95% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

.ai-section {
    width: 100%;
    padding: 20px 24px;
    background: rgba(15,23,42,0.75);
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.35);
    backdrop-filter: blur(6px);
    margin-top: 18px;
}

.ai-section p, .ai-section li {
    font-size: 1.05rem;
    line-height: 1.55rem;
}

.ai-section {
    animation: fadeIn 0.35s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
body {
    background-color: #020617;
    color: #E5E7EB;
}
[data-testid="stAppViewContainer"] {
    background-color: #020617;
}
[data-testid="stSidebar"] {
    background-color: #020617;
}
tbody tr:hover {
    background-color: rgba(8,47,73,0.85) !important;
}
hr.ai-divider {
    border: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #22D3EE, transparent);
    margin: 30px 0;
}

/* Neater, modern tab styling */
.stTabs [data-baseweb="tab"] {
    background: #0F172A;
    color: #CBD5E1;
    padding: 10px 18px;
    border-radius: 8px;
    margin-right: 6px;
    border: 1px solid #1E293B;
    font-weight: 500;
    transition: all 0.15s ease-in-out;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #1E293B;
    color: #F0F9FF;
    border-color: #38BDF8;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #0EA5E9, #38BDF8);
    color: #0F172A !important;
    border-color: #38BDF8;
    font-weight: 600;
    box-shadow: 0 0 12px rgba(56,189,248,0.45);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    position:fixed;
    top:16px;
    right:20px;
    padding:8px 14px;
    background:#020617;
    border-radius:999px;
    border:1px solid #0EA5E9;
    color:#67E8F9;
    font-weight:600;
    font-size:13px;
    box-shadow:0 0 16px rgba(14,165,233,0.45);
    z-index:1000;
">
    🤖 AI Insight Engine
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    padding: 18px 20px;
    border-radius: 16px;
    background: radial-gradient(circle at top left, rgba(56,189,248,0.12), transparent 55%),
                radial-gradient(circle at bottom right, rgba(34,211,238,0.10), transparent 55%),
                #020617;
    border: 1px solid #1F2937;
    box-shadow: 0 0 24px rgba(15,118,110,0.45);
    margin-bottom: 18px;
">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;">
        <div>
            <h1 style="color:#E5E7EB; margin-bottom:4px; font-size:26px;">⚡ Logs Heap AI Analyzer</h1>
            <p style="color:#9CA3AF; margin:0;">
                AI‑powered JVM GC & Application Log Correlation • Observability • Root‑Cause Intelligence
            </p>
        </div>
        <div style="text-align:right; min-width:220px;">
            <div style="color:#67E8F9; font-size:12px; text-transform:uppercase; letter-spacing:0.08em;">
                Active Session
            </div>
            <div style="color:#E5E7EB; font-size:13px;">
                {ts}
            </div>
        </div>
    </div>
</div>
""".format(ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

for key in ["gc_df", "app_df", "correlations", "insight", "ai_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("⚙️ Control Panel")

    model_names = [
        "gpt-oss:120b-cloud",
        "tinyllama:1.1b",
    ]
    if "last_model" not in st.session_state:
        st.session_state.last_model = model_names[0]

    selected_model = st.selectbox(
        "AI Model",
        model_names,
        key="model_selector",
    )

    if selected_model != st.session_state.last_model:
        with st.spinner(f"Switching to model: {selected_model}..."):
            import time
            time.sleep(1)
        st.session_state.last_model = selected_model

    st.markdown("---")
    window_size = st.slider("Correlation Window (seconds)", 1, 30, 5)
    spike_factor = st.slider("GC Spike Threshold (x mean)", 1.0, 3.0, 1.5)

    if st.button("🔄 Reset Session"):
        for key in ["gc_df", "app_df", "correlations", "insight", "ai_mode"]:
            st.session_state[key] = None

        st.session_state.reset_counter = st.session_state.get("reset_counter", 0) + 1
        st.rerun()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 Upload Logs",
    "📄 Parsed Data",
    "🔗 Correlation",
    "🧠 AI Insight",
    "📊 Visualizations",
    "📦 Downloads",
])

# ---------- Tab 1: Upload ----------

with tab1:
    st.subheader("Upload GC and Application Logs")
    st.markdown("""
    <style>
    .upload-card {
        padding: 18px;
        border-radius: 12px;
        background: #0F172A;
        border: 1px solid #1E293B;
        box-shadow: 0 0 12px rgba(56,189,248,0.15);
        transition: 0.2s ease;
        margin-bottom: 12px;
    }
    .upload-card:hover {
        border-color: #38BDF8;
        box-shadow: 0 0 18px rgba(56,189,248,0.35);
    }
    .upload-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .upload-desc {
        font-size: 13px;
        color: #94A3B8;
        margin-bottom: 12px;
    }
    .gc-title { color: #38BDF8; }
    .app-title { color: #38BDF8; }
    </style>

    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-title gc-title">📘 GC Log</div>
            <div class="upload-desc">Upload your JVM Garbage Collection log (.log or .txt)</div>
        """, unsafe_allow_html=True)

        gc_file = st.file_uploader(
            label="",
            type=["log", "txt"],
            key=f"gc_log_{st.session_state.get('reset_counter', 0)}"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-title app-title">📘 Application Log</div>
            <div class="upload-desc">Upload your application runtime log (.log or .txt)</div>
        """, unsafe_allow_html=True)

        app_file = st.file_uploader(
            label="",
            type=["log", "txt"],
            key=f"app_log_{st.session_state.get('reset_counter', 0)}"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    if gc_file or app_file:
        with st.spinner("Parsing logs with JVM-aware parsers..."):
            gc_df = parse_gc_log(gc_file) if gc_file else None
            app_df = parse_app_log(app_file) if app_file else None

        st.session_state.gc_df = gc_df
        st.session_state.app_df = app_df
        st.session_state.correlations = None
        st.session_state.insight = None
        st.session_state.ai_mode = None

        st.success(f"Parsed GC events: {len(gc_df) if gc_df is not None else 0}, "
                   f"App entries: {len(app_df) if app_df is not None else 0}")

        st.markdown('<hr class="ai-divider">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if gc_df is not None and not gc_df.empty:
                st.markdown("#### GC Sample")
                st.dataframe(gc_df.head(10), use_container_width=True)
            else:
                st.info("No GC log uploaded or parsed.")
        with col2:
            if app_df is not None and not app_df.empty:
                st.markdown("#### Application Log Sample")
                st.dataframe(app_df.head(10), use_container_width=True)
            else:
                st.info("No Application log uploaded or parsed.")
    else:
        st.info("Upload at least one log (GC or Application) to begin.")


# ---------- Tab 2: Parsed Data ----------

with tab2:
    st.subheader("Parsed Data")

    gc_df = st.session_state.gc_df
    app_df = st.session_state.app_df

    if (gc_df is None or gc_df.empty) and (app_df is None or app_df.empty):
        st.info("Upload logs in the Upload tab to see parsed data.")
    else:
        total_gc = len(gc_df) if gc_df is not None else 0
        total_app = len(app_df) if app_df is not None else 0

        st.markdown("""
<div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:12px;">
    <div style="
        flex:1; min-width:180px;
        padding:14px 16px;
        background:#020617;
        border-radius:12px;
        border:1px solid #1F2937;
        box-shadow:0 0 16px rgba(34,211,238,0.12);
    ">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:#9CA3AF;">GC Events</div>
        <div style="font-size:24px; color:#E5E7EB; font-weight:600;">{gc}</div>
    </div>
    <div style="
        flex:1; min-width:180px;
        padding:14px 16px;
        background:#020617;
        border-radius:12px;
        border:1px solid #1F2937;
        box-shadow:0 0 16px rgba(34,211,238,0.12);
    ">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:#9CA3AF;">App Log Entries</div>
        <div style="font-size:24px; color:#E5E7EB; font-weight:600;">{app}</div>
    </div>
</div>
""".format(gc=total_gc, app=total_app), unsafe_allow_html=True)

        st.markdown("### 🔍 Quick Highlights")

        if gc_df is not None and not gc_df.empty and app_df is not None and not app_df.empty:
            col_top_gc, col_top_err, col_top_warn = st.columns(3)
        elif gc_df is not None and not gc_df.empty:
            col_top_gc, = st.columns(1)
            col_top_err = col_top_warn = None
        elif app_df is not None and not app_df.empty:
            col_top_err, col_top_warn = st.columns(2)
            col_top_gc = None
        else:
            col_top_gc = col_top_err = col_top_warn = None

        if gc_df is not None and not gc_df.empty and col_top_gc is not None:
            with col_top_gc:
                st.markdown("**Top 10 GC Pauses**")
                top_gc = gc_df.sort_values("pause_ms", ascending=False).head(10)
                st.dataframe(top_gc[["timestamp", "event", "pause_ms", "severity"]], use_container_width=True, height=260)

        if app_df is not None and not app_df.empty and col_top_err is not None:
            with col_top_err:
                st.markdown("**Top 10 ERROR Logs**")
                top_errors = app_df[app_df["level"] == "ERROR"].head(10)
                st.dataframe(top_errors[["timestamp", "level", "message", "category"]], use_container_width=True, height=260)

        if app_df is not None and not app_df.empty and col_top_warn is not None:
            with col_top_warn:
                st.markdown("**Top 10 WARN Logs**")
                top_warns = app_df[app_df["level"] == "WARN"].head(10)
                st.dataframe(top_warns[["timestamp", "level", "message", "category"]], use_container_width=True, height=260)

        st.markdown('<hr class="ai-divider">', unsafe_allow_html=True)

        gc_col, app_col = st.columns(2)
        with gc_col:
            st.markdown("### GC Parsed Data (full)")
            if gc_df is not None and not gc_df.empty:
                st.dataframe(gc_df, use_container_width=True, height=400)
                st.markdown("**GC Severity Breakdown**")
                st.bar_chart(gc_df["severity"].value_counts())
            else:
                st.info("No GC data available.")

        with app_col:
            st.markdown("### Application Parsed Data (full)")
            if app_df is not None and not app_df.empty:
                st.dataframe(app_df, use_container_width=True, height=400)
                st.markdown("**App Log Level Distribution**")
                st.bar_chart(app_df["level"].value_counts())
            else:
                st.info("No Application log data available.")


# ---------- Tab 3: Correlation ----------

with tab3:
    st.subheader("Correlation")

    st.markdown("""
<div style="
    margin-top:8px;
    padding:14px 16px;
    background:rgba(56,189,248,0.08);
    border-left:4px solid #38BDF8;
    border-radius:8px;
    font-size:14px;
">
    <strong style="color:#38BDF8;">What does Correlation mean?</strong>
    <p style="color:#E2E8F0; margin-top:6px;">
        Correlation links GC pauses with application log events that occur within
        your selected time window. If an app error, warning, or slowdown happens
        shortly before or after a GC pause, the analyzer treats them as related.
        Each match is scored based on:
        <ul>
            <li><strong>GC pause severity</strong> (longer pauses score higher)</li>
            <li><strong>App log level</strong> (ERROR > WARN > INFO)</li>
            <li><strong>Time difference</strong> between GC and app event</li>
        </ul>
        Higher scores indicate stronger evidence that a GC event impacted the application.
    </p>
</div>
""", unsafe_allow_html=True)

    gc_df = st.session_state.gc_df
    app_df = st.session_state.app_df

    if gc_df is None or gc_df.empty or app_df is None or app_df.empty:
        st.info("Correlation requires both GC and Application logs.")
    else:
        with st.spinner("Analyzing temporal relationships between GC and App events..."):
            correlations = correlate(gc_df, app_df, window_size, spike_factor)

        st.session_state.correlations = correlations

        if correlations is None or correlations.empty:
            st.warning("No correlations found. Try increasing the window size or check timestamp formats.")
        else:
            st.markdown("### 🚨 Top 10 by Correlation Score")

            top_corr = correlations.sort_values("correlation_score", ascending=False).head(10)

            st.dataframe(
                top_corr[
                    [
                        "gc_timestamp", "gc_event", "gc_pause_ms",
                        "heap_before_k", "heap_after_k", "heap_total_k",
                        "app_timestamp", "app_level", "app_category",
                        "app_message", "time_diff_ms",
                        "correlation_score", "correlation_severity",
                        "correlation_type"
                    ]
                ],
                use_container_width=True,
                height=300,
            )

            st.markdown('<hr class="ai-divider">', unsafe_allow_html=True)

            st.markdown("### ⏱ Top 10 by Time Difference (ms)")

            top_diff = correlations.reindex(
                correlations["time_diff_ms"].abs().sort_values(ascending=False).head(10).index
            )

            st.dataframe(
                top_diff[
                    [
                        "gc_timestamp", "gc_event", "gc_pause_ms",
                        "app_timestamp", "app_level", "app_category",
                        "app_message", "time_diff_ms",
                        "correlation_score", "correlation_severity"
                    ]
                ],
                use_container_width=True,
                height=300,
            )

            st.markdown('<hr class="ai-divider">', unsafe_allow_html=True)

            st.markdown("### 📊 Correlation Severity Distribution")
            st.bar_chart(correlations["correlation_severity"].value_counts())

            st.markdown('<hr class="ai-divider">', unsafe_allow_html=True)

            st.markdown("### 📄 Full Correlation Table")
            st.dataframe(correlations, use_container_width=True, height=400)

            st.markdown('<hr class="ai-divider">', unsafe_allow_html=True)

            st.subheader("🧵 AI Timeline Narration (from Correlations)")

            if st.button("Generate Timeline Based on Current Correlations"):
                with st.spinner("Generating AI narration of correlated events over time..."):
                    timeline_text = ai_timeline_narration(
                        gc_df=gc_df,
                        app_df=app_df,
                        correlations=correlations,
                        model=selected_model,
                    )
                st.markdown(
                    f"""
<div style="margin-top:10px; padding:14px 16px; border-radius:12px;
            background:rgba(34,197,235,0.08); border:1px solid rgba(56,189,248,0.9);">
<pre style="white-space:pre-wrap; font-family:var(--font-mono); font-size:13px; color:#E5E7EB;">{timeline_text}</pre>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.markdown('<hr class="ai-divider">', unsafe_allow_html=True)

            st.markdown("### 🧩 Grouped View (GC Event → App Events)")
            grouped = build_grouped_correlations(correlations)
            for group in grouped:
                with st.expander(
                    f"{group['gc_timestamp']} | {group['gc_event']} "
                    f"({group['gc_pause_ms']} ms, severity={group['severity_max']})"
                ):
                    st.write("Heap:", {
                        "before_k": group["heap_before_k"],
                        "after_k": group["heap_after_k"],
                        "total_k": group["heap_total_k"],
                    })
                    st.write("Correlated App Events:")
                    st.table(pd.DataFrame(group["app_events"]))

# ---------- Tab 4: AI Insight ----------

with tab4:
    st.subheader("AI Insight")

    gc_df = st.session_state.gc_df
    app_df = st.session_state.app_df
    correlations = st.session_state.correlations

    st.markdown("""
<div style="
    margin-top:4px;
    margin-bottom:16px;
    padding:14px 16px;
    background:rgba(15,23,42,0.95);
    border-radius:12px;
    border:1px solid rgba(148,163,184,0.5);
">
  <div style="font-size:13px; color:#E5E7EB; margin-bottom:6px;">
    <strong style="color:#38BDF8;">AI-Driven Analysis Layer</strong><br/>
    Use these focused AI tools to get a narrative, patterns, summaries, and a single best-guess root cause.
  </div>
</div>
""", unsafe_allow_html=True)

    if (gc_df is None or gc_df.empty) and (app_df is None or app_df.empty):
        st.info("Upload at least one log (GC or Application) to generate AI insight.")
    else:
        # --- Improved 4-button horizontal layout ---
        st.markdown("""
    <style>
    .ai-btn {
        background: #0F172A;
        border: 1px solid #1E293B;
        color: #E5E7EB;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: 0.15s ease-in-out;
    }
    .ai-btn:hover {
        background: #1E293B;
        border-color: #38BDF8;
        color: #F0F9FF;
        box-shadow: 0 0 10px rgba(56,189,248,0.4);
    }
    .ai-btn:active {
        background: linear-gradient(90deg, #0EA5E9, #38BDF8);
        color: #0F172A;
        border-color: #38BDF8;
        box-shadow: 0 0 14px rgba(56,189,248,0.6);
    }
    </style>
    """, unsafe_allow_html=True)

    # 4 buttons in one row
    colA, colB, colC, colD = st.columns(4)

    with colA:
        rca_clicked = st.button("🔎 Root Cause", key="btn_rca", help="AI Root-Cause Analyzer", use_container_width=True)

    with colB:
        patterns_clicked = st.button("🧩 Patterns", key="btn_patterns", help="AI Pattern Detector", use_container_width=True)

    with colC:
        timeline_clicked = st.button("🧵 Timeline", key="btn_timeline", help="AI Timeline Narration", use_container_width=True)

    with colD:
        summary_clicked = st.button("🧾 Summaries", key="btn_summary", help="AI Log Summaries", use_container_width=True)

        if gc_df is not None and not gc_df.empty and app_df is not None and not app_df.empty and correlations is not None and not correlations.empty:
            data_for_ai = correlations
            mode = "full"
        elif gc_df is not None and not gc_df.empty:
            data_for_ai = gc_df
            mode = "gc_only"
        elif app_df is not None and not app_df.empty:
            data_for_ai = app_df
            mode = "app_only"
        else:
            st.info("No usable data found for AI insight.")
            data_for_ai = None
            mode = None

        st.session_state.ai_mode = mode

        if data_for_ai is None or mode is None:
            st.info("Please ensure at least one parsed dataset is available.")
        else:
            if st.session_state.insight is None:
                progress = st.progress(0, text="Initializing AI insight pipeline...")

                progress.progress(25, text="Converting data to JSON-safe format...")
                _ = convert(data_for_ai.to_dict(orient="records")) if isinstance(data_for_ai, pd.DataFrame) else convert(data_for_ai)

                progress.progress(50, text=f"Building prompt for model: {selected_model}...")
                progress.progress(75, text="Sending request to AI model...")

                with st.spinner("🤖 AI engine analyzing patterns, correlations, and anomalies..."):
                    insight = generate_ai_insight(data_for_ai, selected_model, mode)

                progress.progress(100, text="AI analysis complete!")
                st.session_state.insight = insight
            else:
                insight = st.session_state.insight

            st.markdown("""
            <div style="display:flex; flex-wrap:wrap; gap:16px; margin-bottom:16px;">
            <div style="flex:1; min-width:260px; padding:14px 16px; border-radius:12px;
                        background:rgba(8,47,73,0.8); border:1px solid rgba(34,211,238,0.4);">
                <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:#7DD3FC;">Mode</div>
                <div style="font-size:18px; color:#E5E7EB; font-weight:600;">{mode}</div>
            </div>
            <div style="flex:2; min-width:260px; padding:14px 16px; border-radius:12px;
                        background:rgba(8,47,73,0.8); border:1px solid rgba(34,211,238,0.4);">
                <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:#7DD3FC;">Root Cause</div>
                <div style="font-size:16px; color:#E5E7EB;">{rc}</div>
            </div>
            <div style="flex:1; min-width:260px; padding:14px 16px; border-radius:12px;
                        background:rgba(8,47,73,0.8); border:1px solid rgba(34,211,238,0.4);">
                <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:#7DD3FC;">Confidence</div>
                <div style="font-size:18px; color:#E5E7EB; font-weight:600;">{conf}</div>
            </div>
            </div>
            """.format(
                mode=mode,
                rc=insight.get('root_cause', ''),
                conf=insight.get('confidence', '')
            ), unsafe_allow_html=True)

            st.markdown(f"""
            <div class="ai-section">
                <h4 style="color:#38BDF8;">Impact</h4>
                <p>{insight.get('impact', '')}</p>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"**Confidence explanation:** {insight.get('confidence_explanation', '')}")

            st.markdown("""
            <div style="
                margin-top:12px;
                padding:14px 16px;
                background:rgba(14,165,233,0.08);
                border-left:4px solid #38BDF8;
                border-radius:8px;
            ">
                <strong style="color:#38BDF8;">What does the Confidence Score mean?</strong>
                <p style="color:#E2E8F0; margin-top:6px;">
                    The confidence score reflects how strongly the AI believes the identified
                    root cause and recommendations match the patterns found in your logs.
                    Higher confidence usually indicates:
                    <ul style="margin-top:4px;">
                        <li>Clear, repeated evidence in GC or App logs</li>
                        <li>Strong correlation between GC pauses and application issues</li>
                        <li>Consistent patterns across timestamps, severity, and categories</li>
                    </ul>
                    Lower confidence typically means the data is noisy, inconsistent, or
                    insufficient for a strong conclusion.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<hr class="ai-divider">', unsafe_allow_html=True)

            st.markdown('<div class="ai-section">', unsafe_allow_html=True)
            st.markdown("""
            <div class="ai-section" style="display: flex; gap: 24px;">
            <div style="flex: 1;">
                <h4 style="color:#38BDF8;">Evidence</h4>
                <ul style="padding-left: 1rem;">
            """, unsafe_allow_html=True)

            for e in insight.get("evidence", [])[:10]:
                st.markdown(f"<li>{e}</li>", unsafe_allow_html=True)

            st.markdown("""
                </ul>
            </div>
            <div style="flex: 1;">
                <h4 style="color:#38BDF8;">Recommendations</h4>
                <ul style="padding-left: 1rem;">
            """, unsafe_allow_html=True)

            for r in insight.get("recommendations", [])[:10]:
                st.markdown(f"<li>{r}</li>", unsafe_allow_html=True)

            st.markdown("""
                </ul>
            </div>
            <div style="flex: 1;">
                <h4 style="color:#38BDF8;">Next Steps</h4>
                <ul style="padding-left: 1rem;">
            """, unsafe_allow_html=True)

            for s in insight.get("next_steps", [])[:10]:
                st.markdown(f"<li>{s}</li>", unsafe_allow_html=True)

            st.markdown("""
                </ul>
            </div>
            </div>
            """, unsafe_allow_html=True)

            

# ---------- Tab 5: Visualizations ----------

with tab5:
    st.subheader("Visualizations")

    gc_df = st.session_state.gc_df
    app_df = st.session_state.app_df
    correlations = st.session_state.correlations

    if (gc_df is None or gc_df.empty) and (app_df is None or app_df.empty):
        st.info("Upload and parse logs first.")
    else:
        st.markdown("### 📊 Session Metrics")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total GC Events", len(gc_df) if gc_df is not None else 0)
        with col_s2:
            st.metric("Total App Logs", len(app_df) if app_df is not None else 0)
        with col_s3:
            st.metric("Correlated Events", len(correlations) if correlations is not None else 0)

        st.markdown('<hr class="ai-divider">', unsafe_allow_html=True)

        if gc_df is not None and not gc_df.empty:
            gc_fig = plot_gc_timeline(gc_df)
            if gc_fig:
                st.plotly_chart(gc_fig, use_container_width=True)

            heatmap_fig = plot_gc_pause_heatmap(gc_df)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)

        if correlations is not None and not correlations.empty:
            corr_fig = plot_correlation_timeline(correlations)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)


# ---------- Tab 6: Downloads ----------

with tab6:
    st.subheader("Download Reports")

    gc_df = st.session_state.gc_df
    app_df = st.session_state.app_df
    correlations = st.session_state.correlations
    insight = st.session_state.insight
    mode = st.session_state.ai_mode or "unknown"

    if (gc_df is None or gc_df.empty) and (app_df is None or app_df.empty):
        st.info("Upload and parse logs first to enable downloads.")
    elif insight is None:
        st.info("Generate AI insight in the AI Insight tab first.")
    else:
        text_report = build_text_report(gc_df, app_df, correlations, insight, mode)
        full_zip = build_full_session_zip(gc_df, app_df, correlations, insight, text_report)
        html_report = build_html_report(gc_df, app_df, correlations, insight, mode)

        st.download_button(
            label="📦 Download Full Session Report (ZIP)",
            data=full_zip,
            file_name=f"jvm_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
        )

        st.download_button(
            label="🌐 Download Full Session Report (HTML)",
            data=html_report.encode("utf-8"),
            file_name=f"jvm_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
        )