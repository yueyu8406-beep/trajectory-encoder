"""Frozen Encoder API for already-local ENU metre coordinates."""
from pathlib import Path
import numpy as np
import torch
from .data import features
from .model import TrajectoryEncoder, pool


class FrozenEncoder:
    def __init__(self, checkpoint, pooling="sum", device="cpu"):
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        self.settings, self.preprocess = state["settings"], state["preprocess"]
        self.device, self.pooling = torch.device(device), pooling
        self.model = TrajectoryEncoder(self.settings["dimension"], **self.settings["model"])
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.model.eval().to(self.device).requires_grad_(False)

    @torch.inference_mode()
    def encode(self, enu_m, resample_from_1hz=True):
        """Return per-token unit vectors and one trajectory vector.

        enu_m: finite (N,3) east/north/up positions in metres relative to RKSI.
        1 Hz input is reduced to every fifth point plus the final endpoint.
        With resample_from_1hz=False the caller supplies the same 5s cadence
        (plus optional final endpoint). Do not pass latitude/longitude here.
        Direction is the forward difference; a final direction is repeated.
        No timestamp/flight-ID or hour-scene inference is performed.
        """
        p = np.asarray(enu_m, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError("Expected at least two finite ENU metre XYZ points")
        if resample_from_1hz:
            step = self.preprocess["preprocessing"]["stride_s"]
            indices = np.arange(0, len(p), step)
            if indices[-1] != len(p) - 1:
                indices = np.append(indices, len(p) - 1)
            p = p[indices]
        if len(p) < self.preprocess["preprocessing"]["min_points"]:
            raise ValueError("Trajectory is shorter than the trained minimum of 30 sampled points")
        x = (features(p) - np.array(self.preprocess["mean"])) / np.array(self.preprocess["std"])
        x = torch.tensor(x, dtype=torch.float32, device=self.device)[None]
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=self.device)
        z = self.model(x, valid)[0]
        return {"tokens": z.cpu().numpy(), "embedding": pool(z, self.pooling).cpu().numpy(),
                "n_points": len(p), "pooling": self.pooling,
                "preprocessor_id": self.preprocess["preprocessor_id"]}
