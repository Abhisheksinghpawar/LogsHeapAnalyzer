import re
import pandas as pd
from dateutil import parser

def parse_gc_log(text):
    patterns = [
        r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{4}).*?\[(?P<event>GC|Full GC).*?,\s*(?P<duration>[\d\.]+)\s*secs\]",
        r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{4}).*?\[(?P<event>G1 Evacuation Pause).*?duration\s*(?P<duration>[\d\.]+)ms",
        r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{4}).*?\[(?P<event>Pause Young).*?(\s|,)(?P<duration>[\d\.]+)ms",
    ]

    rows = []
    for p in patterns:
        for m in re.finditer(p, text):
            ts = parser.parse(m.group("ts"))
            event = m.group("event")
            duration = float(m.group("duration"))
            if "secs" in m.group(0):
                duration *= 1000
            rows.append({"timestamp": ts, "event": event, "duration_ms": duration})

    return pd.DataFrame(rows)