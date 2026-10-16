"""Emit selected publishable UTF-8 source/reports; never reads credentials or raw TSV."""
import json
import sys
from pathlib import Path
root = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")
paths = [p for p in root.iterdir() if p.is_file() and (p.suffix in (".py", ".md", ".json", ".txt") or p.name == ".gitignore")]
paths += list((root / "trajectory_encoder").glob("*.py"))
paths += list((root / "offline_guard").glob("*.py"))
paths += [p for p in (root / "reports").glob("*.json") if p.name != "pipeline_status.json"]
if "--final" in sys.argv:
    for d in ("encoder", "runs"):
        paths += [p for p in (root / d).rglob("*") if p.is_file() and p.suffix in (".json", ".log")]
    paths += list((root / "reports").glob("*.md"))
if "--list" in sys.argv:
    print(json.dumps([str(p.relative_to(root)).replace("\\", "/") for p in paths]))
else:
    print(json.dumps([{"path": str(p.relative_to(root)).replace("\\", "/"), "mode": "100644", "type": "blob", "content": p.read_text(encoding="utf-8-sig")} for p in sorted(set(paths))], ensure_ascii=False))
