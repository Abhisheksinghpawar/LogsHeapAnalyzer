import json
import re
import numpy as np
import pandas as pd
import ollama

# Universal JSON-safe converter
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


# Extract JSON from ANY model output
def extract_json(text):
    if not text or not text.strip():
        return None

    # Find the first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)

    return None


def generate_ai_insight(correlations, model):
    safe = convert(correlations)

    prompt = f"""
You are a JVM performance expert.

Analyze the following correlated GC and application log events.

You MUST respond with ONLY valid JSON.
No markdown. No backticks. No commentary.

JSON schema:
{{
  "root_cause": "",
  "impact": "",
  "evidence": "",
  "recommendations": "",
  "confidence": 0.0
}}

Correlations:
{json.dumps(safe, indent=2)}
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"]

    # Debug print
    print("\nRAW MODEL OUTPUT:\n", raw, "\n")

    json_text = extract_json(raw)

    if not json_text:
        return {
            "root_cause": "Model did not return JSON.",
            "impact": raw[:500] if raw else "No output from model.",
            "evidence": "N/A",
            "recommendations": "Try a smaller model or adjust the prompt.",
            "confidence": 0
        }

    try:
        return json.loads(json_text)
    except Exception:
        return {
            "root_cause": "Model returned invalid JSON.",
            "impact": raw[:500],
            "evidence": "N/A",
            "recommendations": "Try a smaller model or adjust the prompt.",
            "confidence": 0
        }