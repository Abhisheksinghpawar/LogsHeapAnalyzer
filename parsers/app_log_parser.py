import re
import pandas as pd
from dateutil import parser

def parse_app_log(text):
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(INFO|WARN|ERROR)\s+(.*)"
    matches = re.findall(pattern, text)

    rows = []
    for ts, level, msg in matches:
        rows.append({
            "timestamp": parser.parse(ts),
            "level": level,
            "message": msg.strip(),
            "severity": {"INFO": 1, "WARN": 2, "ERROR": 3}[level]
        })

    return pd.DataFrame(rows)