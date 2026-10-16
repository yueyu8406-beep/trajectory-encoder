"""Predeclared length-confounding and route-separability diagnostics; no model selection."""
import numpy as np
import torch
from trajectory_encoder.common import ROOT, read_json, write_json
from trajectory_encoder.data import load_data
from trajectory_encoder.evaluation import probe
from trajectory_encoder.model import TrajectoryEncoder
from trajectory_encoder.train import Batches, embed, seed_all


def knn5(vectors, records):
    split = np.array([r["split"] for r in records])
    names = sorted({r["STAR"] for r in records})
    labels = np.array([names.index(r["STAR"]) for r in records])
    tr, te = [np.flatnonzero(split == s) for s in ("train", "test")]
    z = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    predicted = []
    for chunk in np.array_split(te, max(1, int(np.ceil(len(te)/512)))):
        similarities = z[chunk] @ z[tr].T
        neighbors = np.argpartition(-similarities, 4, axis=1)[:, :5]
        votes = np.zeros((len(chunk), len(names)), dtype=np.int64)
        for j in range(5):
            np.add.at(votes, (np.arange(len(chunk)), labels[tr[neighbors[:, j]]]), 1)
        predicted.extend(votes.argmax(1).tolist())
    predicted = np.asarray(predicted)
    return {"k": 5, "distance": "cosine", "global_test_accuracy": float(np.mean(predicted == labels[te])),
            "test_macro_recall": float(np.mean([np.mean(predicted[labels[te] == i] == i) for i in np.unique(labels[te])])),
            "n_test": len(te), "STARs": names, "tie_rule": "lexicographic STAR among tied vote counts"}


def main():
    cfg = read_json(ROOT / "config.json")
    selection = read_json(ROOT / "reports" / "selection.json")
    arrays, records, _, _ = load_data(ROOT / "prepared")
    lengths = np.array([r["n_points"] for r in records], dtype=np.float64)[:, None]
    lengths /= lengths[[r["split"] == "train" for r in records]].max()
    length_proxy = np.concatenate([lengths, lengths**2, lengths**3], axis=1)
    selected = selection["selected_run"]
    values = np.load(ROOT / "encoder" / "embeddings.npz")
    torch.set_num_threads(2)
    seed_all(selected["seed"])
    model = TrajectoryEncoder(selected["dimension"], **cfg["model"]).cuda()
    data = Batches(arrays, records, selected["epsilon"], "cuda")
    batch = read_json(ROOT / "reports" / "benchmark.json")["selected_batch_size"]
    initial = embed(model, data, batch, ("train", "val", "test"))
    result = {"not_used_for_selection": True, "length_only_cubic_ridge": probe(length_proxy, records, cfg, True),
              "route_knn5": {m: {"trained": knn5(values[m], records), "untrained": knn5(initial[m], records)} for m in values.files},
              "interpretation": "Positive geometry-proxy R2 alone is insufficient: compare random Encoder and length-only baselines; route separability is not operational complexity."}
    write_json(ROOT / "reports" / "confounding_and_route.json", result)
    print(result)


if __name__ == "__main__":
    main()
