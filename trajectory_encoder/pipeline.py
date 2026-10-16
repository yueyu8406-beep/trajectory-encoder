"""Bounded experiment grid; all decisions are fixed before final TEST evaluation."""
import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .common import ROOT, read_json, write_json, code_fingerprint


def run_name(d, e, t, s):
    return f"d{d}_e{int(e)}_t{str(t).replace('.', 'p')}_s{s}"


def complete(job):
    path = ROOT / "runs" / job["run_id"] / "status.json"
    return path.exists() and read_json(path).get("phase") == "complete"


def execute(jobs, batch_size, concurrency):
    pending, running = [j for j in jobs if not complete(j)], []
    while pending or running:
        while pending and len(running) < concurrency:
            job = pending.pop(0)
            directory = ROOT / "runs" / job["run_id"]
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "console.log").open("a", encoding="utf-8")
            args = [sys.executable, "-u", "-m", "trajectory_encoder.train", "--run-id", job["run_id"],
                    "--dimension", str(job["dimension"]), "--epsilon", str(job["epsilon"]),
                    "--temperature", str(job["temperature"]), "--seed", str(job["seed"]), "--batch-size", str(batch_size)]
            process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((process, log, job))
            print(f"START {job['run_id']} pid={process.pid}", flush=True)
        for process, log, job in list(running):
            code = process.poll()
            if code is not None:
                log.close()
                running.remove((process, log, job))
                if code != 0 or not complete(job):
                    for other, _, _ in running:
                        other.terminate()
                    raise RuntimeError(f"Worker failed {job['run_id']}, exit={code}; inspect console.log")
                print(f"COMPLETE {job['run_id']}", flush=True)
        write_json(ROOT / "reports" / "pipeline_status.json", {"running": [j[2]["run_id"] for j in running],
                   "pending": [j["run_id"] for j in pending], "updated_unix": time.time()})
        if pending or running:
            time.sleep(5)


def main():
    cfg = read_json(ROOT / "config.json")
    benchmark = read_json(ROOT / "reports" / "benchmark.json")
    batch = benchmark["selected_batch_size"]
    concurrency = min(benchmark["max_parallel_workers"], cfg["training"]["max_parallel_training_workers"])
    screening = [{"run_id": run_name(8, e, t, 42), "dimension": 8, "epsilon": e, "temperature": t, "seed": 42}
                 for e, t in itertools.product(cfg["preprocessing"]["rdp_epsilon_m"], cfg["training"]["temperatures"])]
    write_json(ROOT / "reports" / "frozen_protocol.json", {"stage": "before_training", "code_sha256": code_fingerprint(),
        "screening": screening, "batch_size": batch, "concurrency": concurrency,
        "checkpoint_rule": "minimum validation flat-SNN loss within each run",
        "hyperparameter_rule": "maximum validation geometric-proxy ridge R2 mean across four targets, global and within-STAR, sum pooling",
        "dimension_rule": "maximum three-seed mean of same validation score, sum pooling; lower dimension breaks exact ties",
        "representative_seed_rule": "median validation-score seed of selected dimension",
        "pooling_rule": "sum is main; mean and signed GeM-p2 are sensitivity analyses",
        "TEST": "only after all choices are frozen; no TEST-driven revisions"})
    execute(screening, batch, concurrency)
    def score(job):
        return read_json(ROOT / "runs" / job["run_id"] / "validation.json")["sum"]["probes"]["validation_selection_score"]
    chosen = max(screening, key=score)
    write_json(ROOT / "reports" / "screening_selection.json", {"scores": [{**j, "validation_score": score(j)} for j in screening],
                                                               "chosen": chosen, "selection_data": "validation only"})
    grid = [{"run_id": run_name(d, chosen["epsilon"], chosen["temperature"], s), "dimension": d,
             "epsilon": chosen["epsilon"], "temperature": chosen["temperature"], "seed": s}
            for d, s in itertools.product(cfg["training"]["dimensions"], cfg["training"]["seeds"])]
    execute(grid, batch, concurrency)
    averages = {str(d): float(np.mean([score(j) for j in grid if j["dimension"] == d])) for d in cfg["training"]["dimensions"]}
    dimension = max(sorted(cfg["training"]["dimensions"]), key=lambda d: averages[str(d)])
    selected = sorted([j for j in grid if j["dimension"] == dimension], key=score)[1]
    selection = {"hyperparameters": chosen, "dimension_validation_means": averages, "selected_dimension": dimension,
                 "selected_run": selected, "main_pooling": "sum", "representative_seed": "median validation score",
                 "grid": [{**j, "validation_score": score(j)} for j in grid], "frozen_before_TEST": True,
                 "test_not_loaded_for_selection": True, "frozen_unix": time.time()}
    write_json(ROOT / "reports" / "selection.json", selection)
    print(f"FROZEN SELECTION {selected}", flush=True)


if __name__ == "__main__":
    main()
