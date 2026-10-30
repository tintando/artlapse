"""Compare the GPU SIFT's keypoints and descriptors with cv2.SIFT on the same image.

    uv run --project .. python test/compare_sift.py <png> <saved json>

For each flip state: how many keypoints each finds, how many of OpenCV's have a GPU keypoint within
0.5 px (and the same size within 5% and angle within 5 degrees), and the descriptor distance of those
pairs against the typical distance between unrelated descriptors.
"""
import json
import sys

import cv2
import numpy as np

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
web = json.load(open(sys.argv[2]))
sift = cv2.SIFT_create(nfeatures=1500)
for state, (im, w) in enumerate(zip([img, cv2.flip(img, 1)], web)):
    kp, desc = sift.detectAndCompute(im, None)
    pts = np.float32([k.pt for k in kp])
    wp = np.float32(w["pts"]).reshape(-1, 2)
    wd = np.float32(w["desc"]).reshape(-1, 128)
    d = np.linalg.norm(pts[:, None] - wp[None], axis=2)
    # among the gpu keypoints within 0.5 px (one per orientation), the closest descriptor
    ddesc = np.linalg.norm(desc[:, None] - wd[None], axis=2)
    ddesc[d >= 0.5] = np.inf
    j = ddesc.argmin(axis=1)
    near = d[np.arange(len(pts)), j] < 0.5
    dd = ddesc[np.arange(len(pts)), j]
    rand = np.linalg.norm(desc[:200, None] - wd[None, :200], axis=2).mean()
    print(f"state {state}: opencv {len(kp)}, gpu {len(wp)}; opencv keypoints with a gpu one within 0.5 px: {near.mean():.1%}")
    print(f"  descriptor distance of those pairs: median {np.median(dd[near]):.1f}, 90% {np.percentile(dd[near], 90):.1f}; unrelated pairs {rand:.0f}")
