"""Local-only launcher; disables shell telemetry and Python outbound sockets.

Use a command shell without PowerShell startup telemetry. Environment changes
apply only to this process tree and are not persisted to the user's machine.
"""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
env = os.environ.copy()
env.update({"POWERSHELL_TELEMETRY_OPTOUT": "1", "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "PYTHONPATH": str(root / "offline_guard"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true", "PYTHONUNBUFFERED": "1"})
stage = sys.argv[1] if len(sys.argv) > 1 else "probe"
if stage == "probe":
    code = """
import socket, time
for operation in (lambda: socket.getaddrinfo('example.com',443), lambda: socket.socket().connect(('1.1.1.1',443))):
    try:
        operation()
    except PermissionError:
        print('PASS: outbound attempt was rejected before connection', flush=True)
    else:
        raise RuntimeError('Offline guard failed')
import torch
assert torch.cuda.is_available()
x=torch.randn(128,128,device='cuda'); print('CUDA local probe',float((x@x.T).mean()),flush=True)
for i in range(7):
    time.sleep(10); print('offline probe heartbeat',i+1,flush=True)
print('OFFLINE PROBE COMPLETE',flush=True)
"""
    command = [sys.executable, "-u", "-c", code]
elif stage in ("pipeline", "finalize", "checks", "benchmark"):
    command = [sys.executable, "-u", "-m", "trajectory_encoder." + stage]
else:
    raise ValueError("Supported: probe, pipeline, finalize, checks, benchmark")
raise SystemExit(subprocess.call(command, cwd=root, env=env))
