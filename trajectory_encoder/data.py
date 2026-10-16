"""Fresh ATFMTraj preprocessing. No old files, checkpoints or caches are read."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, digest, file_hash, read_json, write_json

FEATURES = ["east_m", "north_m", "up_m", "direction_e", "direction_n", "direction_u", "radius_m", "sin_theta", "cos_theta"]


def features(p):
    delta = np.diff(p, axis=0)
    delta = np.concatenate([delta, delta[-1:]])
    unit = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
    theta = np.arctan2(p[:, 1], p[:, 0])
    return np.column_stack([p, unit, np.linalg.norm(p[:, :2], axis=1), np.sin(theta), np.cos(theta)])


def rdp(p, epsilon):
    keep = np.zeros(len(p), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(p) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        line, v = p[b] - p[a], p[a + 1:b] - p[a]
        norm = np.linalg.norm(line)
        distances = (np.linalg.norm(v, axis=1) if norm < 1e-12 else np.linalg.norm(np.cross(v, line), axis=1) / norm)
        i = int(np.argmax(distances))
        if distances[i] > epsilon:
            mid = a + 1 + i
            keep[mid] = True
            stack.extend([(a, mid), (mid, b)])
    labels = np.cumsum(keep).astype(np.int16) - 1
    labels[-1] = labels[-2]
    return labels


def geometry(p, times):
    delta = np.diff(p, axis=0)
    norm = np.linalg.norm(delta, axis=1)
    units = delta / np.maximum(norm[:, None], 1e-9)
    ok = (norm[:-1] > 1e-9) & (norm[1:] > 1e-9)
    angle = np.arccos(np.clip(np.sum(units[:-1] * units[1:], axis=1)[ok], -1, 1))
    length = float(norm.sum())
    return {"path_m": length, "curvature_rad": float(angle.sum()),
            "sinuosity": length / max(float(np.linalg.norm(p[-1] - p[0])), 1.0),
            "duration_s": int(times[-1] - times[0])}


def signature(p, count, grid_m):
    grid = np.linspace(0, len(p) - 1, count)
    sampled = np.column_stack([np.interp(grid, np.arange(len(p)), p[:, j]) for j in range(3)])
    return hashlib.sha256(np.rint(sampled / grid_m).astype(np.int32).tobytes()).hexdigest()


def prepare(source, destination, cfg):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("Fresh preparation requires a new destination")
    pcfg = cfg["preprocessing"]
    stats = pd.read_csv(source / "RKSIa_stats.csv")
    route_names = {int(row.Class): str(row.Description).split(",")[0].strip()
                   for row in stats.itertuples() if str(row.Class).isdigit()}
    rows, source_info, padding_total = [], [], 0
    for mode in ("TRAIN", "TEST"):
        channels, classes = [], None
        for axis in "XYZ":
            path = source / f"RKSIa_{mode}_{axis}.tsv"
            a = pd.read_csv(path, sep="\t", header=None, dtype=np.float64).to_numpy()
            if classes is None:
                classes = a[:, 0].astype(np.int64)
            elif not np.array_equal(classes, a[:, 0]):
                raise ValueError("X/Y/Z labels are not aligned")
            channels.append(a[:, 1:])
            source_info.append({"filename": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size,
                                "rows": len(a), "padded_samples_per_row": a.shape[1] - 1})
            print(f"read {path.name}: {a.shape}", flush=True)
        if len({a.shape for a in channels}) != 1:
            raise ValueError("XYZ shape mismatch")
        xyz = np.stack(channels, axis=-1)
        del channels, a
        for i, raw in enumerate(xyz):
            finite = np.isfinite(raw)
            if np.any(finite.any(1) != finite.all(1)):
                raise ValueError(f"Partial XYZ missingness: {mode}/{i}")
            valid = finite.all(1)
            n = int(valid.sum())
            if not np.array_equal(valid, np.arange(len(valid)) < n):
                raise ValueError(f"Non-trailing NaN: {mode}/{i}")
            padding_total += int((~valid).sum())
            full = np.ascontiguousarray(raw[:n])
            exact = hashlib.sha256(full.tobytes()).hexdigest()
            indices = np.arange(0, n, pcfg["stride_s"])
            if n and indices[-1] != n - 1:
                indices = np.append(indices, n - 1)
            p = full[indices] * pcfg["r_max_m"]
            if len(p) < pcfg["min_points"]:
                continue
            c = int(classes[i])
            rows.append({"p": p, "time": indices, "partition": mode, "row": i,
                         "exact": exact, "group": signature(p, pcfg["near_duplicate_signature_points"], pcfg["near_duplicate_grid_m"]),
                         "class_id": c, "STAR": route_names.get(c, str(c)), "raw_n": n})
        del xyz
    test_groups = {r["group"] for r in rows if r["partition"] == "TEST"}
    test_exact = {r["exact"] for r in rows if r["partition"] == "TEST"}
    clean, seen, excluded = [], set(), {"exact_duplicates": 0, "train_test_overlap": 0}
    for r in rows:
        if r["partition"] == "TRAIN" and (r["group"] in test_groups or r["exact"] in test_exact):
            excluded["train_test_overlap"] += 1
            continue
        if r["exact"] in seen:
            excluded["exact_duplicates"] += 1
            continue
        seen.add(r["exact"])
        clean.append(r)
    groups = sorted({r["group"] for r in clean if r["partition"] == "TRAIN"})
    np.random.default_rng(pcfg["split_seed"]).shuffle(groups)
    val_groups = set(groups[:max(2, int(round(len(groups) * pcfg["validation_fraction_of_official_train"])) )])
    records, xx, pp, times, labels = [], [], [], [], {str(int(e)): [] for e in pcfg["rdp_epsilon_m"]}
    for r in clean:
        split = "test" if r["partition"] == "TEST" else "val" if r["group"] in val_groups else "train"
        p, t = r["p"], r["time"]
        records.append({"trajectory_id": f"RKSIa:{r['partition']}:{r['row']}", "source_partition": r["partition"],
                        "source_row": r["row"], "split": split, "class_id": r["class_id"], "STAR": r["STAR"],
                        "group_id": r["group"], "coordinate_sha256": r["exact"], "raw_n_points": r["raw_n"],
                        "n_points": len(p), **geometry(p, t)})
        xx.append(features(p))
        pp.append(p.astype(np.float32))
        times.append(t.astype(np.int32))
        for epsilon in pcfg["rdp_epsilon_m"]:
            labels[str(int(epsilon))].append(rdp(p, epsilon))
    counts = {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")}
    train_x = np.concatenate([x for x, r in zip(xx, records) if r["split"] == "train"])
    mean, std = train_x.mean(0), train_x.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    del train_x
    normalized = np.concatenate([((x - mean) / std).astype(np.float32) for x in xx])
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite features")
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in pp])])
    spec = {"feature_order": FEATURES, "mean": mean.tolist(), "std": std.tolist(),
            "preprocessing": pcfg, "normalization": "zscore_train_tokens_only", "airport": "RKSI",
            "absolute_timestamps_available": False, "direction": "forward_difference_repeat_final",
            "RDP_units": "metres_before_feature_standardization", "time_basis": "relative_elapsed_seconds"}
    spec["preprocessor_id"] = digest(spec)
    fingerprint = digest({"sources": source_info, "records": records, "preprocessor_id": spec["preprocessor_id"]})
    audit = {"dataset": "ATFMTraj_RKSIa", "sources": source_info, "tracks": len(records), "splits": counts,
             "excluded": excluded, "missing_XYZ_inside_tracks": 0, "trailing_padding_triplets": padding_total,
             "length_min_median_max": np.quantile(np.diff(offsets), [0, .5, 1]).tolist(),
             "rdp_segment_length_quantiles": {e: np.quantile(np.concatenate([np.bincount(a) for a in labs]), [0, .5, 1]).tolist() for e, labs in labels.items()},
             "fingerprint": fingerprint, "preprocessor_id": spec["preprocessor_id"],
             "group_rule": "full-coordinate exact duplicate + 32-point/50m quantized shape signature",
             "limitations": ["No dates, flight IDs, or absolute timestamps: true date/flight identity grouping cannot be verified.",
                             "Shape signatures are a conservative partial overlap screen, not proof that all near duplicates are detected.",
                             "Official TEST is not used to fit normalization or model/probe parameters.",
                             "No real hourly traffic scenes can be reconstructed from these TSV files."]}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / "arrays.npz", x=normalized, p=np.concatenate(pp), time=np.concatenate(times), offsets=offsets,
                        **{"labels_" + e: np.concatenate(v) for e, v in labels.items()})
    for name, obj in (("metadata", records), ("preprocess", spec), ("audit", audit)):
        write_json(destination / f"{name}.json", obj)
    print(f"prepared {len(records)} trajectories: {counts}", flush=True)


def load_data(directory):
    directory = Path(directory)
    with np.load(directory / "arrays.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    return arrays, read_json(directory / "metadata.json"), read_json(directory / "preprocess.json"), read_json(directory / "audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default=str(ROOT / "prepared"))
    args = parser.parse_args()
    prepare(args.source, args.out, read_json(ROOT / "config.json"))
