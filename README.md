📘 Logs Heap AI Analyzer
AI‑powered JVM Observability & Log Intelligence Console

Logs Heap AI Analyzer is a full‑stack JVM observability tool that ingests GC logs, application logs, correlates them intelligently, visualizes performance patterns, and generates AI‑driven root‑cause analysis — all inside a clean, Grafana‑style engineering console.
This tool is designed for JVM engineers, SREs, backend developers, and performance analysts who want clarity, speed, and actionable insights from massive log files.

🏷️ Badges
    

🎥 Demo (GIF Placeholder)


If you want, I can generate a storyboard for the GIF so you can record it cleanly.

🚀 Key Features
🔍 1. GC Log Parsing (JDK‑style logs)
• 	Supports GC, Full GC, Metadata GC Threshold, Allocation Failure, Ergonomics events
• 	Extracts:
• 	Timestamp
• 	Pause duration
• 	Heap before/after
• 	GC category
• 	Severity scoring
• 	Timezone‑safe parsing (handles  offsets)

📄 2. Application Log Parsing
• 	Supports standard JVM app logs:

• 	Extracts:
• 	Timestamp
• 	Log level (INFO/WARN/ERROR)
• 	Message
• 	Category (Database, Network, Memory, General)

🔗 3. GC ↔ App Log Correlation Engine
A custom correlation engine that matches GC events with app events using:
• 	Time‑window matching
• 	Pause severity scoring
• 	Log level weighting
• 	Time‑difference scoring
• 	Correlation severity classification
• 	Grouped correlation view (GC event → related app events)
This produces high‑signal, low‑noise correlation insights.

📊 4. Plotly Visualizations
Interactive, Grafana‑style charts:
• 	GC Timeline (pause duration over time)
• 	GC Pause Heatmap (per minute, per category)
• 	Correlation Timeline (GC vs App events)
All charts are zoomable, hoverable, and exportable.

🧠 5. AI‑Powered Insight Generation
Uses an LLM to produce:
• 	Root cause
• 	Impact summary
• 	Evidence list
• 	Recommendations
• 	Next steps
• 	Confidence score
• 	Confidence explanation
The model receives a JSON‑safe correlation dataset and returns structured JSON.

📦 6. Full Session Export
Downloadable artifacts include:
ZIP Report
• 	Parsed GC CSV
• 	Parsed App CSV
• 	Correlations JSON
• 	AI Insight JSON
• 	AI Insight TXT report
HTML Report (Dashboard‑Style)
A single self‑contained HTML file with:
• 	Summary & Highlights (first tab)
• 	Top 10 GC pauses
• 	Top 10 ERROR logs
• 	Top 10 correlations
• 	AI root cause, evidence, recommendations
• 	GC Details (full table)
• 	App Log Details (full table)
• 	Correlations (full table)
• 	AI Insight (raw JSON)
This is perfect for sharing with teams, attaching to tickets, or archiving.

🖥 7. Engineering Console UI (Streamlit)
A clean, AWS CloudWatch / Grafana‑style layout with tabs:
Tab 1 — Upload Logs
• 	Upload GC + App logs
• 	Shows sample rows
Tab 2 — Parsed Data
Summary Panels (Top 10):
• 	GC pauses
• 	ERROR logs
• 	WARN logs
Then full tables + severity charts.
Tab 3 — Correlation
Summary Panels (Top 10):
• 	By correlation score
• 	By time difference
Then:
• 	Severity distribution
• 	Full correlation table
• 	Grouped correlation view
Tab 4 — AI Insight
Summary Panel:
• 	Root cause
• 	Impact
• 	Confidence
• 	Top 10 evidence
• 	Top recommendations
• 	Next steps
Then raw JSON.
Tab 5 — Visualizations
• 	Quick stats
• 	GC timeline
• 	GC heatmap
• 	Correlation timeline
Tab 6 — Downloads
• 	ZIP report
• 	HTML report

⚡ 8. Large‑Dataset Friendly
Even with huge logs, the UI stays usable because:
• 	Summary panels show only the most important 10 rows
• 	Full tables are scrollable
• 	Correlation engine is optimized
• 	Timezone normalization prevents mismatches
• 	No giant tables at the top of any tab
This keeps the user focused on insights, not noise.

🛠 Installation


🚦 Quickstart Guide
1. Launch the app

2. Upload logs
• 	
• 	
3. Review parsed data
• 	GC events
• 	App logs
• 	Summary panels
4. Correlate
• 	GC ↔ App event matching
• 	Severity scoring
• 	Top 10 insights
5. Generate AI Insight
• 	Root cause
• 	Evidence
• 	Recommendations
6. Visualize
• 	GC timeline
• 	Heatmap
• 	Correlation timeline
7. Export
• 	ZIP report
• 	HTML dashboard report

🧩 Troubleshooting
GC timestamps not matching app timestamps
Ensure GC logs contain timezone offsets ().
The parser normalizes them automatically.
No correlations found
Try increasing:
• 	Correlation window (seconds)
• 	Spike factor
AI insight fails
Try switching to a smaller model in the sidebar.
Large logs slow down UI
Use summary panels — they’re designed for this.

❓ FAQ
Does this work with huge logs?
Yes — summary panels prevent UI overload, and the correlation engine is optimized.
Can I use custom AI models?
Yes — any Ollama‑compatible model works.
Can I export charts?
Plotly charts support built‑in export.
Can I embed this in CI/CD?
Yes — the HTML report is perfect for automated pipelines.

🤝 Contributing
Contributions are welcome!
• 	Fork the repo
• 	Create a feature branch
• 	Submit a PR
• 	Add tests where possible
If you want, I can also generate:
• 	A CONTRIBUTING.md
• 	A CODE_OF_CONDUCT.md

📜 License
MIT License
Feel free to use, modify, and distribute.