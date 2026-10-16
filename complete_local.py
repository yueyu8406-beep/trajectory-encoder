"""Resume the frozen training grid and finish local evaluation with no outbound networking."""
import os
import argparse
from pathlib import Path
import subprocess
import sys
import time
import json

root = Path(__file__).resolve().parent
env = os.environ.copy()
env.update({"POWERSHELL_TELEMETRY_OPTOUT": "1", "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "PYTHONPATH": str(root / "offline_guard"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true", "PYTHONUNBUFFERED": "1"})
stages = [("training", ["-m", "trajectory_encoder.pipeline"]),
          ("final_test_and_export", ["-m", "trajectory_encoder.finalize"]),
          ("confounding_diagnostics", ["posthoc_diagnostics.py"]),
          ("report", ["make_report.py"]),
          ("verify_delivery", ["verify_delivery.py"])]
parser = argparse.ArgumentParser()
parser.add_argument("--start-at", choices=[name for name, _ in stages], default="training")
options = parser.parse_args()
stages = stages[[name for name, _ in stages].index(options.start_at):]
for name, args in stages:
    status = {"stage": name, "started_unix": time.time(), "state": "running"}
    (root / "reports" / "completion_status.json").write_text(json.dumps(status), encoding="utf-8")
    print("STAGE", name, flush=True)
    subprocess.run([sys.executable, "-u", *args], cwd=root, env=env, check=True)
(root / "reports" / "completion_status.json").write_text(json.dumps({"state": "complete", "finished_unix": time.time()}), encoding="utf-8")
print("ALL LOCAL TRAINING, EVALUATION AND DELIVERY CHECKS COMPLETE", flush=True)
