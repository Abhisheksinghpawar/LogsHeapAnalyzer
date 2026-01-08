import plotly.express as px
import plotly.graph_objects as go

def plot_gc_timeline(gc_df):
    return px.line(gc_df, x="timestamp", y="duration_ms", markers=True)

def plot_log_heatmap(app_df, gc_df):
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=app_df["timestamp"],
        y=["Severity"] * len(app_df),
        z=app_df["severity"],
        colorscale="RdYlGn_r"
    ))

    for ts in gc_df["timestamp"]:
        fig.add_vline(x=ts, line_width=1, line_color="blue")

    return fig