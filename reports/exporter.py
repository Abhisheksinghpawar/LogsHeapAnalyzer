import json

def generate_json_report(insight):
    return json.dumps(insight, indent=2)

def generate_markdown_report(insight):
    md = "# JVM Intelligence Report\n\n"
    for key, value in insight.items():
        md += f"## {key.replace('_', ' ').title()}\n{value}\n\n"
    return md