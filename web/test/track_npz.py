"""Write a web track (test/register.ts's .track.json) as artlapse's track.npz, into a new directory, for
artlapse check and the other tools: track_npz.py <track.json> <new dir>"""
import json
import sys
from pathlib import Path

import numpy as np

d = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
out.mkdir(parents=True)
n = len(d["status"])
np.savez_compressed(out / "track.npz", fps=d["fps"], status=np.uint8(d["status"]), m=np.float64(d["m"]).reshape(n, 3, 3),
                    inliers=np.int32(d["inliers"]))
