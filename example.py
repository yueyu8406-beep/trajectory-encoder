"""python example.py --checkpoint encoder/encoder.pt --input your_enu.npy"""
import argparse
import numpy as np
from trajectory_encoder.inference import FrozenEncoder

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="encoder/encoder.pt")
p.add_argument("--input", default="encoder/example_input_output.npz")
p.add_argument("--one-hz", action="store_true", help="Input is 1 Hz instead of already sampled 5s ENU")
a = p.parse_args()
value = np.load(a.input)
points = value["enu_m"] if isinstance(value, np.lib.npyio.NpzFile) else value
result = FrozenEncoder(a.checkpoint).encode(points, resample_from_1hz=a.one_hz)
print("token vectors:", result["tokens"].shape)
print("trajectory vector:", result["embedding"])
