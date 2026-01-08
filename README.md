📘 Logs Heap AI Analyzer
AI‑powered JVM GC + Application Log Correlation & Insight Console
A fast, intelligent, and developer‑friendly tool for analyzing JVM GC logs and application logs together — with AI‑generated root‑cause analysis, interactive visualizations, and exportable reports.

🚀 Key Features (At a Glance)

🔍 GC Log Parsing
- Supports GC, Full GC, Metadata GC Threshold, Allocation Failure
- Extracts timestamps, pause times, heap before/after, GC category, severity
- Timezone‑safe parsing

📄 Application Log Parsing
- Supports standard JVM log format
- Extracts timestamp, level, message, category (DB, Network, Memory, General)

🔗 Correlation Engine
- Matches GC events with app logs using time‑window matching
- Scores severity using pause time, log level, and time difference
- Grouped correlation view (GC event → related app events)

📊 Interactive Visualizations
- GC Timeline (pause duration over time)
- GC Heatmap (per minute, per category)
- Correlation Timeline (GC vs App events)

🧠 AI Insight Generation
- Root cause
- Impact summary
- Evidence list
- Recommendations
- Next steps
- Confidence score

📦 Exportable Reports
- HTML Report (Summary + tabs for GC, App, Correlations, AI Insight)
- ZIP Report (CSV, JSON, TXT)

🖥 Streamlit Engineering Console

Tabs include:
- Upload Logs
- Parsed Data (with Top‑10 summary panels)
- Correlation (Top‑10 insights + full table)
- AI Insight
- Visualizations
- Downloads

⚡ Optimized for Large Logs
- Summary panels prevent UI overload
- Scrollable tables
- Efficient correlation logic
- Clean, dashboard‑style layout

🛠 Installation

pip install -r requirements.txt
streamlit run app.py

🚦 Quickstart

- Upload gc.log and app.log
- Review parsed data + Top‑10 highlights
- View correlations
- Generate AI insight
- Explore visualizations
- Export HTML or ZIP report

📜 License

MIT License
