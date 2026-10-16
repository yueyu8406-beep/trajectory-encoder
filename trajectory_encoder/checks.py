"""Mathematical and end-to-end checks for this research implementation."""
import math
import tempfile
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, read_json, write_json
from .model import TrajectoryEncoder, flat_snn, pool
from .data import load_data, features, rdp


def run():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cfg = read_json(ROOT / "config.json")
    model = TrajectoryEncoder(8, **cfg["model"]).eval()
    x, valid = torch.randn(2, 13, 9), torch.ones((2, 13), dtype=torch.bool)
    baseline = model(x, valid)
    changed = x.clone()
    changed[:, 7:] += torch.randn_like(changed[:, 7:]) * 100
    torch.testing.assert_close(baseline[:, :7], model(changed, valid)[:, :7], atol=1e-6, rtol=1e-5)
    mask = torch.zeros_like(valid)
    mask[:, :5] = True
    masked_a, masked_b = model(x, valid, mask), model(changed, valid, mask)
    torch.testing.assert_close(masked_a[:, :5], masked_b[:, :5], atol=1e-6, rtol=1e-5)
    assert torch.isfinite(masked_a).all()
    torch.testing.assert_close(baseline.norm(dim=-1), torch.ones_like(valid, dtype=torch.float), atol=1e-6, rtol=1e-5)
    pad_x = torch.cat([x, torch.randn(2, 4, 9)], 1)
    pad_valid = torch.cat([valid, torch.zeros((2, 4), dtype=torch.bool)], 1)
    torch.testing.assert_close(model(pad_x, pad_valid)[:, :13], baseline, atol=1e-6, rtol=1e-5)
    assert (model(pad_x, pad_valid)[:, 13:] == 0).all()
    z = torch.randn(2, 4, 5, requires_grad=True)
    labels = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    v = torch.ones((2, 4), dtype=torch.bool)
    loss, anchors = flat_snn(z, labels, v, .05)
    e = torch.nn.functional.normalize(z.flatten(0, 1), dim=-1)
    ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    manual = []
    for i in range(8):
        positive = torch.stack([e[i] @ e[j] / .05 for j in range(8) if j != i and ids[j] == ids[i]])
        negative = torch.stack([e[i] @ e[j] / .05 for j in range(8) if ids[j] != ids[i]])
        manual.append(torch.logsumexp(negative, 0) - torch.logsumexp(positive, 0))
    torch.testing.assert_close(loss, torch.stack(manual).mean())
    grad = torch.autograd.grad(loss, z, retain_graph=True)[0]
    chunked, _ = flat_snn(z, labels, v, .05, full_limit=1, chunk_size=3)
    torch.testing.assert_close(loss, chunked)
    torch.testing.assert_close(grad, torch.autograd.grad(chunked, z)[0], atol=1e-5, rtol=1e-5)
    singleton_labels = torch.tensor([[0, 0, 1, 2], [0, 0, 1, 2]])
    singleton_loss, n = flat_snn(z, singleton_labels, v, .01)
    assert n == 4 and torch.isfinite(singleton_loss)
    tiny = torch.tensor([[-2., 1.], [1., -3.]])
    expected = torch.tensor([-math.sqrt(1.5), -2.])
    torch.testing.assert_close(pool(tiny, "gem2"), expected)
    p = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [2., 2., 0.]])
    assert np.array_equal(rdp(p, .1), [0, 0, 1, 1])
    assert features(p).shape == (4, 9) and np.isfinite(features(p)).all()
    arrays, records, pre, audit = load_data(ROOT / "prepared")
    groups = {s: {r["group_id"] for r in records if r["split"] == s} for s in ("train", "val", "test")}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
    train_rows = [arrays["x"][arrays["offsets"][i]:arrays["offsets"][i+1]] for i, r in enumerate(records) if r["split"] == "train"]
    train_features = np.concatenate(train_rows)
    np.testing.assert_allclose(train_features.mean(0, dtype=np.float64), 0., atol=1e-6)
    np.testing.assert_allclose(train_features.std(0, dtype=np.float64), 1., atol=1e-6)
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        path = Path(tmp) / "checkpoint.pt"
        torch.save(model.state_dict(), path)
        restored = TrajectoryEncoder(8, **cfg["model"]).eval()
        restored.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(restored(x, valid), baseline)
    gpu = model.cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = gpu(x.cuda(), valid.cuda(), mask.cuda())
    assert torch.isfinite(output).all()
    result = {"passed": True, "checks": ["Transformer causal prefix", "all-masked prefix has no future-key leakage", "padding invariance", "token L2 norm", "flat-SNN analytical reference", "chunked/full loss value and gradient", "singleton exclusion", "signed GeM equation", "RDP segment identity", "split group disjointness", "train-only z-score", "checkpoint reload", "CUDA bf16 finite output"],
              "device": torch.cuda.get_device_name(), "data_sha256": audit["fingerprint"], "note": "Forward-difference features themselves use the next sampled point, as required by PLAN.md; causality check applies to the Transformer on supplied features."}
    write_json(ROOT / "reports" / "implementation_checks.json", result)
    print(result)


if __name__ == "__main__":
    run()
