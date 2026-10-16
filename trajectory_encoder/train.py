"""Fresh training worker with real minibatches and resumable optimizer/RNG state."""
import argparse
import csv
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, read_json, write_json, code_fingerprint, file_hash
from .data import load_data
from .model import TrajectoryEncoder, flat_snn
from .evaluation import probe, diagnostics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Batches:
    def __init__(self, arrays, records, epsilon, device):
        self.arrays, self.records, self.device = arrays, records, device
        self.offsets = arrays["offsets"]
        self.lengths = np.diff(self.offsets)
        self.labels = arrays["labels_" + str(int(epsilon))]
        self.indices = {s: np.array([i for i, r in enumerate(records) if r["split"] == s]) for s in ("train", "val", "test")}

    def indices_batches(self, split, batch_size, seed=0, shuffle=False):
        indices = self.indices[split].copy()
        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(indices)
        # Random length buckets reduce padding without changing positive/negative definitions.
        buckets = []
        for part in np.array_split(indices, max(1, int(np.ceil(len(indices) / (batch_size * 16))))):
            part = part[np.argsort(self.lengths[part], kind="stable")]
            buckets.extend([part[i:i+batch_size] for i in range(0, len(part), batch_size)])
        if shuffle:
            rng.shuffle(buckets)
        for ids in buckets:
            # A final single trajectory still has many RDP segments and valid negatives.
            yield ids

    def make(self, ids):
        n = int(self.lengths[ids].max())
        x = np.zeros((len(ids), n, 9), dtype=np.float32)
        labels, valid = np.zeros((len(ids), n), dtype=np.int64), np.zeros((len(ids), n), dtype=bool)
        for j, i in enumerate(ids):
            a, b = self.offsets[i:i+2]
            x[j, :b-a], labels[j, :b-a], valid[j, :b-a] = self.arrays["x"][a:b], self.labels[a:b], True
        return tuple(torch.from_numpy(a).to(self.device) for a in (x, labels, valid))


@torch.inference_mode()
def embed(model, data, batch_size, splits=("train", "val")):
    model.eval()
    count, dim = len(data.records), model.post_proj.out_features
    output = {mode: np.zeros((count, dim), dtype=np.float32) for mode in ("sum", "mean", "gem2")}
    for split in splits:
        for ids in data.indices_batches(split, batch_size):
            x, _, valid = data.make(ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x, valid)
            n = valid.sum(1, keepdim=True)
            values = {"sum": z.sum(1), "mean": z.sum(1) / n}
            signed_power = (z.sign() * z.square()).sum(1) / n
            values["gem2"] = signed_power.sign() * signed_power.abs().sqrt()
            for mode, value in values.items():
                output[mode][ids] = value.cpu().numpy()
    return output


def save_checkpoint(path, obj):
    temporary = path.with_suffix(".writing")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def train(run_id, dimension, epsilon, temperature, seed, batch_size, epochs=None):
    cfg = read_json(ROOT / "config.json")
    torch.set_num_threads(cfg["training"]["cpu_threads_per_worker"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_all(seed)
    directory = ROOT / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    arrays, records, pre, audit = load_data(ROOT / "prepared")
    data = Batches(arrays, records, epsilon, "cuda")
    model = TrajectoryEncoder(dimension, **cfg["model"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    settings = {"run_id": run_id, "dimension": dimension, "epsilon_m": epsilon, "temperature": temperature,
                "seed": seed, "batch_size": batch_size, "amp": "bf16", "epochs": epochs or cfg["training"]["epochs"],
                "model": cfg["model"], "training": cfg["training"], "code_sha256": code_fingerprint(),
                "data_sha256": audit["fingerprint"], "preprocessor_id": pre["preprocessor_id"],
                "checkpoint_selection": "minimum deterministic validation flat-SNN loss, all epochs",
                "device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "numpy": np.__version__}
    write_json(directory / "settings.json", settings)
    start_epoch, best, best_epoch = 1, float("inf"), 0
    history = []
    if (directory / "last.pt").exists():
        state = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise ValueError("Resume settings or code fingerprint changed")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        best, best_epoch, start_epoch = state["best"], state["best_epoch"], state["epoch"] + 1
        history = read_json(directory / "history.json")
    elif not (directory / "untrained_validation.json").exists():
        initial = embed(model, data, batch_size)
        write_json(directory / "untrained_validation.json", {m: probe(v, records, cfg) for m, v in initial.items()})
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, settings["epochs"] + 1):
        epoch_start = time.time()
        model.train()
        sums, count, grad_max = 0., 0, 0.
        for ids in data.indices_batches("train", batch_size, seed * 1000 + epoch, True):
            x, labels, valid = data.make(ids)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x, valid)
                loss, anchors = flat_snn(z, labels, valid, temperature, cfg["training"]["full_loss_token_limit"], cfg["training"]["loss_chunk_size"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss {run_id}/{epoch}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            sums += float(loss.detach()) * anchors
            count += anchors
            grad_max = max(grad_max, float(gradient))
        model.eval()
        val_sum, val_count = 0., 0
        with torch.inference_mode():
            for ids in data.indices_batches("val", batch_size):
                x, labels, valid = data.make(ids)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(x, valid)
                    loss, anchors = flat_snn(z, labels, valid, temperature, cfg["training"]["full_loss_token_limit"], cfg["training"]["loss_chunk_size"])
                val_sum += float(loss) * anchors
                val_count += anchors
        val = val_sum / val_count
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss")
        if val < best:
            best, best_epoch = val, epoch
            save_checkpoint(directory / "best.pt", {"state_dict": model.state_dict(), "settings": settings, "epoch": epoch, "validation_loss": val, "preprocess": pre})
        row = {"epoch": epoch, "train_loss": sums/count, "validation_loss": val, "best_epoch": best_epoch,
               "seconds": time.time() - epoch_start, "gradient_norm_max_before_clip": grad_max,
               "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20}
        history.append(row)
        write_json(directory / "history.json", history)
        write_json(directory / "status.json", {"phase": "training", **row, "updated_unix": time.time()})
        save_checkpoint(directory / "last.pt", {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "settings": settings,
                        "epoch": epoch, "best": best, "best_epoch": best_epoch, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()})
        print(f"{run_id} epoch {epoch:02d}/{settings['epochs']} train={row['train_loss']:.5f} val={val:.5f} best={best_epoch} {row['seconds']:.1f}s", flush=True)
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    embeddings = embed(model, data, batch_size)
    np.savez_compressed(directory / "validation_embeddings.npz", **embeddings)
    scores = {m: {"probes": probe(v, records, cfg), "diagnostics": diagnostics(v, records, cfg, "val")} for m, v in embeddings.items()}
    write_json(directory / "validation.json", scores)
    write_json(directory / "status.json", {"phase": "complete", "epochs": settings["epochs"], "best_epoch": best_epoch,
               "best_validation_loss": best, "worker_elapsed_s": time.time() - started,
               "best_checkpoint_sha256": file_hash(directory / "best.pt"), "updated_unix": time.time()})
    print(f"COMPLETE {run_id}: validation score {scores['sum']['probes']['validation_selection_score']:.6f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--dimension", type=int, required=True)
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--epochs", type=int)
    a = p.parse_args()
    train(a.run_id, a.dimension, a.epsilon, a.temperature, a.seed, a.batch_size, a.epochs)
