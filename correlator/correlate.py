import pandas as pd

def correlate_events(gc_df, app_df, window_seconds=5, spike_factor=1.5):
    if gc_df.empty or app_df.empty:
        return []

    threshold = gc_df["duration_ms"].mean() * spike_factor
    spikes = gc_df[gc_df["duration_ms"] >= threshold]

    correlations = []
    for _, spike in spikes.iterrows():
        start = spike["timestamp"] - pd.Timedelta(seconds=window_seconds)
        end = spike["timestamp"] + pd.Timedelta(seconds=window_seconds)

        window_logs = app_df[(app_df["timestamp"] >= start) & (app_df["timestamp"] <= end)]

        correlations.append({
            "gc_event": spike.to_dict(),
            "related_logs": window_logs.to_dict(orient="records"),
            "severity_score": window_logs["severity"].sum()
        })

    return correlations