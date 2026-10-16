"""Package verified Encoder deliverables without raw data or training caches."""
import json
from pathlib import Path
import zipfile
from trajectory_encoder.common import ROOT, read_json, write_json, file_hash

if read_json(ROOT / "reports" / "completion_status.json")["state"] != "complete":
    raise RuntimeError("Training, final evaluation and export checks must finish first")
top = ["README.md", "PLAN.md", "PROTOCOL.md", "CITATION.md", "THIRD_PARTY_LICENSE_ATSCC.md", "config.json", "requirements.txt", ".gitignore",
       "example.py", "run_offline.py", "complete_local.py", "posthoc_diagnostics.py", "make_report.py", "verify_delivery.py", "package_delivery.py"]
paths = [ROOT / name for name in top]
excluded = {"blob_receipts.json", "early_export_manifest.json", "pipeline_status.json", "completion_status.json", "artifact_transfer_checks.json", "repository_delivery.json", "delivery_archive.json"}
for directory in ("trajectory_encoder", "offline_guard", "encoder", "checkpoints", "reports"):
    paths += [p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts
              and p.suffix != ".pyc" and p.name not in excluded]
for directory in (ROOT / "runs").iterdir():
    paths += [p for p in directory.iterdir() if p.is_file() and p.suffix in (".json", ".log")]
archive = ROOT / "encoder_delivery.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as output:
    for p in sorted(set(paths)):
        relative = p.relative_to(ROOT)
        archive_name = (Path("experiment_logs") / Path(*relative.parts[1:])).as_posix() if relative.parts[0] == "runs" else relative.as_posix()
        if p.name == "public_data_verification.json":
            value = read_json(p)
            value.pop("user_publication_instruction", None)
            output.writestr(archive_name, json.dumps(value, indent=2))
        else:
            output.write(p, archive_name)
result = {"filename": archive.name, "bytes": archive.stat().st_size, "sha256": file_hash(archive), "files": len(set(paths)),
          "raw_TSV_included": False, "optimizer_states_included": False}
write_json(ROOT / "reports" / "delivery_archive.json", result)
print(result)
