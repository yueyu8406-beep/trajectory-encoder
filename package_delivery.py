"""Create a self-contained trained Encoder bundle without raw TSV or optimizer caches."""
from pathlib import Path
import zipfile
from trajectory_encoder.common import ROOT, write_json, file_hash

required = [ROOT / "encoder" / "encoder.pt", ROOT / "reports" / "test_results.json", ROOT / "reports" / "export_verification.json"]
if not all(p.exists() for p in required):
    raise RuntimeError("Final training and export must be complete before packaging")
paths = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py", ".md", ".txt", ".json") and p.name not in ("artifact_manifest.json",)]
for directory in ("trajectory_encoder", "offline_guard", "encoder", "checkpoints", "reports"):
    paths += [p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts
              and p.suffix not in (".pyc",) and p.name not in ("blob_receipts.json", "pipeline_status.json")]
for directory in (ROOT / "runs").iterdir():
    paths += [p for p in directory.iterdir() if p.is_file() and p.suffix in (".json", ".log")]
archive = ROOT / "encoder_delivery.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as output:
    for p in sorted(set(paths)):
        output.write(p, p.relative_to(ROOT).as_posix())
result = {"filename": archive.name, "bytes": archive.stat().st_size, "sha256": file_hash(archive), "files": len(set(paths)),
          "raw_TSV_included": False, "optimizer_states_included": False}
write_json(ROOT / "reports" / "delivery_archive.json", result)
print(result)
