"""Train-fitted ridge probes. Validation selects alpha; TEST never selects anything."""
import numpy as np


def r2(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    denominator = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ((y - pred) ** 2).sum(0) / np.maximum(denominator, 1e-12)


def fit_predict(train_x, train_y, val_x, test_x, alphas, val_y):
    mean, std = train_x.mean(0), train_x.std(0)
    std = np.where(std < 1e-8, 1., std)
    x = (train_x - mean) / std
    vx, tx = (val_x - mean) / std, (test_x - mean) / std
    ym, ys = train_y.mean(0), train_y.std(0)
    ys = np.where(ys < 1e-8, 1., ys)
    y = (train_y - ym) / ys
    gram, rhs = x.T @ x, x.T @ y
    candidates = [np.linalg.solve(gram + float(a) * np.eye(x.shape[1]), rhs) for a in alphas]
    vp = np.stack([(vx @ w) * ys + ym for w in candidates])
    scores = np.stack([r2(val_y, pred) for pred in vp])
    chosen = scores.argmax(0)
    weights = np.column_stack([candidates[a][:, j] for j, a in enumerate(chosen)])
    return (vx @ weights) * ys + ym, (tx @ weights) * ys + ym, [alphas[i] for i in chosen]


def probe(embeddings, records, cfg, include_test=False):
    targets = cfg["evaluation"]["targets"]
    y = np.asarray([[r[t] for t in targets] for r in records], dtype=np.float64)
    x = np.asarray(embeddings, dtype=np.float64)
    split, group = np.array([r["split"] for r in records]), np.array([r["STAR"] for r in records])
    tr, va, te = [np.flatnonzero(split == s) for s in ("train", "val", "test")]
    if not include_test:
        te = va[:0]
    vp, tp, alphas = fit_predict(x[tr], y[tr], x[va], x[te], cfg["evaluation"]["ridge_alphas"], y[va])
    result = {"global_validation_R2": dict(zip(targets, r2(y[va], vp).tolist())), "global_alpha": dict(zip(targets, alphas))}
    if include_test:
        result["global_test_R2"] = dict(zip(targets, r2(y[te], tp).tolist()))
    by_group = {}
    for name in np.unique(group):
        gt, gv, ge = [idx[group[idx] == name] for idx in (tr, va, te)]
        if len(gt) < cfg["evaluation"]["minimum_group_train"] or len(gv) < cfg["evaluation"]["minimum_group_validation"]:
            continue
        pval, ptest, alpha = fit_predict(x[gt], y[gt], x[gv], x[ge], cfg["evaluation"]["ridge_alphas"], y[gv])
        entry = {"n_train": len(gt), "n_val": len(gv), "n_test": len(ge), "alpha": dict(zip(targets, alpha)),
                 "validation_R2": dict(zip(targets, r2(y[gv], pval).tolist()))}
        if include_test and len(ge) >= cfg["evaluation"]["minimum_group_test"]:
            entry["test_R2"] = dict(zip(targets, r2(y[ge], ptest).tolist()))
        by_group[str(name)] = entry
    result["within_STAR"] = by_group
    result["within_STAR_validation_macro_R2"] = {t: float(np.mean([g["validation_R2"][t] for g in by_group.values()])) for t in targets}
    if include_test:
        result["within_STAR_test_macro_R2"] = {t: float(np.mean([g["test_R2"][t] for g in by_group.values() if "test_R2" in g])) for t in targets}
    result["validation_selection_score"] = float(np.mean(list(result["global_validation_R2"].values()) + list(result["within_STAR_validation_macro_R2"].values())))
    return result


def diagnostics(embeddings, records, cfg, split_name):
    idx = [i for i, r in enumerate(records) if r["split"] == split_name]
    x = np.asarray(embeddings[idx], dtype=np.float64)
    centered = x - x.mean(0)
    values = np.linalg.eigvalsh(centered.T @ centered / max(len(x) - 1, 1)).clip(0)
    probabilities = values / max(values.sum(), 1e-30)
    effective = float(np.exp(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30)))))
    norm, length = np.linalg.norm(x, axis=1), np.array([records[i]["n_points"] for i in idx])
    rng = np.random.default_rng(2026)
    a, b = rng.integers(0, len(x), size=(2, cfg["evaluation"]["pair_sample_limit"]))
    distances = np.linalg.norm(x[a] - x[b], axis=1)[a != b]
    strata = []
    q = np.quantile(length, [0, 1/3, 2/3, 1])
    for j in range(3):
        mask = (length >= q[j]) & ((length <= q[j+1]) if j == 2 else (length < q[j+1]))
        strata.append({"n": int(mask.sum()), "length_lower": float(q[j]), "length_upper": float(q[j+1]),
                       "norm_mean": float(norm[mask].mean()), "variance_mean": float(x[mask].var(0).mean())})
    return {"n": len(x), "finite": bool(np.isfinite(x).all()), "variance_per_dimension": x.var(0).tolist(),
            "effective_rank": effective, "dimension": x.shape[1],
            "collapsed_dimensions": int((x.var(0) < cfg["evaluation"]["collapse_variance_threshold"]).sum()),
            "pair_distance_quantiles": np.quantile(distances, [0, .05, .5, .95, 1]).tolist(),
            "norm_length_pearson": float(np.corrcoef(norm, length)[0, 1]), "length_strata": strata}
