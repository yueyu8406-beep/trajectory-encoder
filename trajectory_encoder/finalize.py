"""Final held-out evaluation and portable Encoder bundle after frozen selection."""
import csv
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, read_json, write_json, file_hash
from .data import load_data
from .model import TrajectoryEncoder
from .train import Batches, embed, seed_all
from .evaluation import probe, diagnostics, fit_predict, r2
from .inference import FrozenEncoder


def length_stratified(embeddings, records, cfg):
    split = np.array([r["split"] for r in records])
    n = np.array([r["n_points"] for r in records])
    targets = cfg["evaluation"]["targets"]
    y = np.array([[r[t] for t in targets] for r in records])
    tr, va, te = [np.flatnonzero(split == s) for s in ("train", "val", "test")]
    vp, tp, alpha = fit_predict(embeddings[tr], y[tr], embeddings[va], embeddings[te], cfg["evaluation"]["ridge_alphas"], y[va])
    cuts = np.quantile(n[tr], [1/3, 2/3])
    groups = np.digitize(n[te], cuts)
    return {"cuts_from_training_n_points": cuts.tolist(), "test_strata": [
        {"stratum": j, "n": int((groups == j).sum()), "R2": dict(zip(targets, r2(y[te][groups == j], tp[groups == j]).tolist()))}
        for j in range(3)]}


def main():
    cfg = read_json(ROOT / "config.json")
    selection = read_json(ROOT / "reports" / "selection.json")
    assert selection["frozen_before_TEST"]
    torch.set_num_threads(2)
    arrays, records, pre, audit = load_data(ROOT / "prepared")
    batch = read_json(ROOT / "reports" / "benchmark.json")["selected_batch_size"]
    results, initial_results, manifest = {}, {}, []
    export = ROOT / "encoder"
    export.mkdir(exist_ok=True)
    weights = ROOT / "checkpoints"
    weights.mkdir(exist_ok=True)
    selected_run = selection["selected_run"]["run_id"]
    for job in selection["grid"]:
        name = job["run_id"]
        checkpoint = torch.load(ROOT / "runs" / name / "best.pt", map_location="cpu", weights_only=False)
        model = TrajectoryEncoder(job["dimension"], **cfg["model"]).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        data = Batches(arrays, records, job["epsilon"], "cuda")
        values = embed(model, data, batch, ("train", "val", "test"))
        np.savez_compressed(ROOT / "runs" / name / "all_embeddings.npz", **values)
        metrics = {m: {"probes": probe(v, records, cfg, True), "diagnostics": diagnostics(v, records, cfg, "test"),
                       "length_stratified": length_stratified(v, records, cfg)} for m, v in values.items()}
        results[name] = metrics
        seed_all(job["seed"])
        initial = TrajectoryEncoder(job["dimension"], **cfg["model"]).cuda()
        initial_values = embed(initial, data, batch, ("train", "val", "test"))
        initial_results[name] = {m: probe(v, records, cfg, True) for m, v in initial_values.items()}
        portable = {**checkpoint, "state_dict": {k: v.cpu() for k, v in checkpoint["state_dict"].items()}}
        torch.save(portable, weights / (name + ".pt"))
        manifest.append({"run_id": name, "dimension": job["dimension"], "seed": job["seed"],
                         "best_epoch": checkpoint["epoch"], "path": "checkpoints/" + name + ".pt",
                         "bytes": (weights / (name + ".pt")).stat().st_size, "sha256": file_hash(weights / (name + ".pt"))})
        if name == selected_run:
            shutil.copyfile(weights / (name + ".pt"), export / "encoder.pt")
            np.savez_compressed(export / "embeddings.npz", **values)
            write_json(export / "config.json", {"run_id": name, **checkpoint["settings"], "pooling": "sum", "pooling_post_L2": False,
                                                 "best_epoch": checkpoint["epoch"], "selection": selection})
            write_json(export / "preprocess.json", pre)
        write_json(ROOT / "reports" / "test_results.json", results)
        write_json(ROOT / "reports" / "untrained_test_results.json", initial_results)
        print(f"FINAL TEST {name}", flush=True)
        del model, initial
        torch.cuda.empty_cache()
    write_json(ROOT / "reports" / "checkpoint_manifest.json", manifest)
    # Verify the user-facing loader on real prepared coordinates and exact weights.
    loader = FrozenEncoder(export / "encoder.pt", device="cpu")
    i = next(i for i, r in enumerate(records) if r["split"] == "test")
    a, b = arrays["offsets"][i:i+2]
    encoded = loader.encode(arrays["p"][a:b], resample_from_1hz=False)
    expected_model = TrajectoryEncoder(selection["selected_dimension"], **cfg["model"]).eval()
    checkpoint = torch.load(export / "encoder.pt", map_location="cpu", weights_only=False)
    expected_model.load_state_dict(checkpoint["state_dict"])
    with torch.inference_mode():
        z = expected_model(torch.from_numpy(arrays["x"][a:b])[None], torch.ones((1, b-a), dtype=torch.bool))[0].numpy()
    error = float(np.max(np.abs(z - encoded["tokens"])))
    assert error < 1e-4, error
    assert not any(p.requires_grad for p in loader.model.parameters())
    np.savez_compressed(export / "example_input_output.npz", enu_m=arrays["p"][a:b], **{k: encoded[k] for k in ("tokens", "embedding")})
    write_json(ROOT / "reports" / "export_verification.json", {"passed": True, "strict_state_dict_load": True,
        "frozen_parameters": True, "repeated_feature_rebuild_max_token_error": error,
        "example_trajectory_id": records[i]["trajectory_id"], "shape_tokens": list(encoded["tokens"].shape),
        "shape_embedding": list(encoded["embedding"].shape), "selected_weights_sha256": file_hash(export / "encoder.pt")})
    # Traceable sample split and metadata without redistributing the raw TSV data.
    with (ROOT / "reports" / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    shutil.copyfile(ROOT / "prepared" / "audit.json", ROOT / "reports" / "data_audit.json")
    print("FINAL EVALUATION AND EXPORT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
