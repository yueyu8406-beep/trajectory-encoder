"""Measure actual B=8/16 mixed-precision steps, including the longest tracks."""
import time
import numpy as np
import torch
from .common import ROOT, read_json, write_json
from .data import load_data
from .model import TrajectoryEncoder, flat_snn
from .train import Batches, seed_all


def run():
    cfg = read_json(ROOT / "config.json")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(42)
    arrays, records, _, _ = load_data(ROOT / "prepared")
    data = Batches(arrays, records, 120, "cuda")
    results = []
    for b in cfg["training"]["batch_candidates"]:
        model = TrajectoryEncoder(8, **cfg["model"]).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
        ids = list(data.indices_batches("train", b, 42, True))
        longest = data.indices["train"][np.argsort(data.lengths[data.indices["train"]])[-b:]]
        trial = ids[:12] + [longest]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        try:
            for j, rows in enumerate(trial):
                x, labels, valid = data.make(rows)
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = flat_snn(model(x, valid), labels, valid, .01,
                                      cfg["training"]["full_loss_token_limit"], cfg["training"]["loss_chunk_size"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
            result = {"batch_size": b, "success": True, "mean_regular_step_seconds": float(np.mean(durations[2:-1])),
                      "estimated_train_epoch_seconds": float(np.mean(durations[2:-1]) * len(ids)),
                      "longest_batch_step_seconds": durations[-1], "peak_allocated_MiB": torch.cuda.max_memory_allocated()/2**20,
                      "peak_reserved_MiB": torch.cuda.max_memory_reserved()/2**20, "longest_tokens": int(data.lengths[longest].sum())}
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            result = {"batch_size": b, "success": False, "reason": "CUDA OOM"}
        print(result, flush=True)
        results.append(result)
        del model, optimizer
        torch.cuda.empty_cache()
    successful = [r for r in results if r["success"]]
    selected = min(successful, key=lambda r: r["estimated_train_epoch_seconds"])
    result = {"device": torch.cuda.get_device_name(), "total_MiB": torch.cuda.get_device_properties(0).total_memory/2**20,
              "AMP": "bf16", "gradient_accumulation": False, "trials": results, "selected_batch_size": selected["batch_size"],
              "max_parallel_workers": 2 if selected["peak_reserved_MiB"] * 2 + 1100 < torch.cuda.get_device_properties(0).total_memory/2**20 else 1}
    write_json(ROOT / "reports" / "benchmark.json", result)


if __name__ == "__main__":
    run()
