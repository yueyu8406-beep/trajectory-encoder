"""Verify downloaded checkpoint bytes and the ready-to-use Encoder example."""
from pathlib import Path
import numpy as np
from trajectory_encoder.common import ROOT, file_hash, read_json
from trajectory_encoder.inference import FrozenEncoder

manifest = read_json(ROOT / "reports" / "checkpoint_manifest.json")
for entry in manifest:
    checkpoint = ROOT / entry["path"]
    if file_hash(checkpoint) != entry["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {checkpoint.name}")
    print("SHA256 OK", entry["path"])
spec = read_json(ROOT / "reports" / "export_verification.json")
assert file_hash(ROOT / "encoder" / "encoder.pt") == spec["selected_weights_sha256"]
example = np.load(ROOT / "encoder" / "example_input_output.npz")
encoder = FrozenEncoder(ROOT / "encoder" / "encoder.pt")
out = encoder.encode(example["enu_m"], resample_from_1hz=False)
np.testing.assert_allclose(out["tokens"], example["tokens"], rtol=1e-5, atol=1e-6)
np.testing.assert_allclose(out["embedding"], example["embedding"], rtol=1e-5, atol=1e-5)
print("FROZEN ENCODER EXAMPLE OK", out["tokens"].shape, out["embedding"].shape)
