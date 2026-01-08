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
    return obj


# ---------- GC log parser ----------

def parse_gc_log(file) -> pd.DataFrame:
    text = file.read().decode("utf-8", errors="ignore")
    lines = [l for l in text.splitlines() if l.strip()]
    records = []

    for line in lines:
        # Example:
        # 2024-11-18T10:15:32.123+0000: 1.234: [GC (Allocation Failure) 512K->256K(1024K), 0.0056789 secs]
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
        # Example:
        # 2024-11-18 10:15:31 INFO  Starting request handling for /api/user
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
    # Convert DataFrame → list of dicts → JSON-safe values
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
      body {{ padding: 20px; }}
      .tab-content {{ margin-top: 20px; }}
      pre {{ white-space: pre-wrap; }}
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
    fig.update_layout(xaxis_title="Time", yaxis_title="Pause (ms)")
    return fig


def plot_correlation_timeline(corr_df: pd.DataFrame):
    if corr_df is None or corr_df.empty:
        return None
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(corr_df["gc_timestamp"]),
        y=[0] * len(corr_df),
        mode="markers",
        marker=dict(size=10, color="red"),
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
        marker=dict(size=9, color="blue"),
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
    )
    return fig


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Logs Heap AI Analyzer", layout="wide")
st.title("Logs Heap AI Analyzer")

for key in ["gc_df", "app_df", "correlations", "insight", "ai_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("⚙️ Settings")

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Upload Logs",
    "Parsed Data",
    "Correlation",
    "AI Insight",
    "Visualizations",
    "Downloads",
])

# ---------- Tab 1: Upload ----------

with tab1:
    st.subheader("Upload GC and Application Logs")

    gc_file = st.file_uploader("GC Log", type=["log", "txt"], key="gc_log")
    app_file = st.file_uploader("Application Log", type=["log", "txt"], key="app_log")

    if gc_file or app_file:
        with st.spinner("Parsing logs..."):
            gc_df = parse_gc_log(gc_file) if gc_file else None
            app_df = parse_app_log(app_file) if app_file else None

        st.session_state.gc_df = gc_df
        st.session_state.app_df = app_df
        st.session_state.correlations = None
        st.session_state.insight = None
        st.session_state.ai_mode = None

        st.success(f"Parsed GC events: {len(gc_df) if gc_df is not None else 0}, "
                   f"App entries: {len(app_df) if app_df is not None else 0}")

        col1, col2 = st.columns(2)
        with col1:
            if gc_df is not None and not gc_df.empty:
                st.markdown("**Sample GC rows**")
                st.dataframe(gc_df.head(10), use_container_width=True)
            else:
                st.info("No GC log uploaded or parsed.")
        with col2:
            if app_df is not None and not app_df.empty:
                st.markdown("**Sample App rows**")
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
        st.info("Upload logs in the first tab to see parsed data.")
    else:
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

        st.markdown("---")

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

    gc_df = st.session_state.gc_df
    app_df = st.session_state.app_df

    if gc_df is None or gc_df.empty or app_df is None or app_df.empty:
        st.info("Correlation requires both GC and Application logs.")
    else:
        with st.spinner("Computing correlations..."):
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

            st.markdown("---")

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

            st.markdown("---")

            st.markdown("### 📊 Correlation Severity Distribution")
            st.bar_chart(correlations["correlation_severity"].value_counts())

            st.markdown("---")

            st.markdown("### 📄 Full Correlation Table")
            st.dataframe(correlations, use_container_width=True, height=400)

            st.markdown("---")

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

    if (gc_df is None or gc_df.empty) and (app_df is None or app_df.empty):
        st.info("Upload at least one log (GC or Application) to generate AI insight.")
    else:
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
                progress = st.progress(0, text="Preparing AI analysis...")

                progress.progress(25, text="Converting data to JSON-safe format...")
                _ = data_for_ai.to_dict(orient="records") if isinstance(data_for_ai, pd.DataFrame) else convert(data_for_ai)

                progress.progress(50, text=f"Building prompt for model: {selected_model}...")
                progress.progress(75, text="Sending request to AI model...")

                with st.spinner("AI is analyzing JVM behavior and application impact..."):
                    insight = generate_ai_insight(data_for_ai, selected_model, mode)

                progress.progress(100, text="AI analysis complete!")
                st.session_state.insight = insight
            else:
                insight = st.session_state.insight

            st.markdown("### 🧠 Key Findings")

            st.write(f"**Mode:** {mode}")
            st.write(f"**Root cause:** {insight.get('root_cause', '')}")
            st.write(f"**Impact:** {insight.get('impact', '')}")
            st.write(f"**Confidence:** {insight.get('confidence', '')}")
            st.write(f"**Confidence explanation:** {insight.get('confidence_explanation', '')}")

            col_e, col_r, col_n = st.columns(3)

            with col_e:
                st.markdown("**Top Evidence (up to 10)**")
                for e in insight.get("evidence", [])[:10]:
                    st.markdown(f"- {e}")

            with col_r:
                st.markdown("**Recommendations**")
                for r in insight.get("recommendations", [])[:10]:
                    st.markdown(f"- {r}")

            with col_n:
                st.markdown("**Next steps**")
                for s in insight.get("next_steps", [])[:10]:
                    st.markdown(f"- {s}")

            st.markdown("---")
            st.markdown("### Raw Insight JSON")
            st.json(insight)


# ---------- Tab 5: Visualizations ----------

with tab5:
    st.subheader("Visualizations")

    gc_df = st.session_state.gc_df
    app_df = st.session_state.app_df
    correlations = st.session_state.correlations

    if (gc_df is None or gc_df.empty) and (app_df is None or app_df.empty):
        st.info("Upload and parse logs first.")
    else:
        st.markdown("### 📊 Quick Stats")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total GC Events", len(gc_df) if gc_df is not None else 0)
        with col_s2:
            st.metric("Total App Logs", len(app_df) if app_df is not None else 0)
        with col_s3:
            st.metric("Correlated Events", len(correlations) if correlations is not None else 0)

        st.markdown("---")

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
            label="Download Full Session Report (ZIP)",
            data=full_zip,
            file_name=f"jvm_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
        )

        st.download_button(
            label="Download Full Session Report (HTML)",
            data=html_report.encode("utf-8"),
            file_name=f"jvm_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
        )