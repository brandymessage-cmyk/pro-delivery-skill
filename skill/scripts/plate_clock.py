"""Plate as clock: measures for every take frame which (fractional) frame of the original plate it matches, and builds
a retime curve from that which aligns the take to the plate frame by frame.

python plate_clock.py take.mp4 plate.mov --frames 119 --out tau [--search 8] [--snap 0.12] [--sigma 1.0] [--tol 0.3] [--pad 2]

Compares only where the plate moves (motion mask from the plate), 960 px, luma, normalised (lighting and
colour differences do not count). Output: tau.npy (raw), tau_curve.npy (smoothed; where the curve deviates less than `snap`
from a whole frame, it is snapped to it = real frame, no in-between), tau_nuke.txt (Kronos
`timing2 Frame`, Nuke counts from 1) and a summary. Requirement: take and plate show the same action with a
locked camera. Then: in-betweens only for the fractional values (Kronos for rigid objects, RIFE for limbs)."""
import os, sys, argparse, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bg_pulse import lumas

a = argparse.ArgumentParser(); a.add_argument("take"); a.add_argument("plate"); a.add_argument("--frames", type=int, required=True)
a.add_argument("--out", default="tau"); a.add_argument("--search", type=int, default=8)
a.add_argument("--snap", type=float, default=0.05); a.add_argument("--sigma", type=float, default=1.0)
a.add_argument("--tol", type=float, default=0.3); a.add_argument("--pad", type=int, default=2)
o = a.parse_args()
P = np.stack([cv2.GaussianBlur(x, (0, 0), 1.2) for x in lumas(o.plate)]); T = np.stack([cv2.GaussianBlur(x, (0, 0), 1.2) for x in lumas(o.take)])
mot = np.abs(np.diff(P, axis=0)).max(0)
mask = cv2.dilate((cv2.GaussianBlur(mot, (0, 0), 3) > 0.03).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
print(f"Plate {len(P)} frames, take {len(T)} frames, motion mask {mask.mean() * 100:.1f} % of the image")
nz = lambda x: (x[mask] - x[mask].mean()) / (x[mask].std() + 1e-6)
Pn = np.stack([nz(p) for p in P]); Tn = np.stack([nz(t) for t in T])
tau, cost = [], []
for k in range(len(T)):
    js = np.arange(max(0, k - o.search), min(len(P) - 1, k + o.search) + 1)
    d = np.array([np.mean((Tn[k] - Pn[j]) ** 2) for j in js]); i = int(np.argmin(d)); off = 0.0
    if 0 < i < len(js) - 1:
        den = d[i - 1] - 2 * d[i] + d[i + 1]; off = float(np.clip(0.5 * (d[i - 1] - d[i + 1]) / den, -0.5, 0.5)) if den > 1e-9 else 0.0
    tau.append(js[i] + off); cost.append(float(d[i]))
tau = np.array(tau); np.save(f"{o.out}.npy", tau)
# inversion: for every output frame j (= plate frame j) the take position s whose tau = j
tm = np.maximum.accumulate(cv2.GaussianBlur(tau.reshape(-1, 1), (0, 0), o.sigma).ravel())
j = np.arange(o.frames); s = np.interp(j, tm, np.arange(len(T)))
ok = np.flatnonzero((j >= tm[2]) & (j <= tm[-3]))          # edges: measurement unreliable -> offset from the nearest reliable frame
s[:ok[0]] = j[:ok[0]] + round(s[ok[0]] - j[ok[0]]); s[ok[-1] + 1:] = j[ok[-1] + 1:] + round(s[ok[-1]] - j[ok[-1]])
s = np.clip(s, 0, len(T) - 1)
# A constant offset (e.g. take one frame ahead) needs no in-betweens: real frames round(s) there.
# In-betweens only where the curve lies clearly between two frames (> tol), gaps up to 6 frames closed,
# extended by `pad` frames so that entry and exit are soft.
bad = np.abs(s - np.round(s)) > o.tol; idx = np.flatnonzero(bad)
for x, y in zip(idx[:-1], idx[1:]):
    if y - x <= 6: bad[x:y] = True
for _ in range(o.pad): bad = bad | np.r_[bad[1:], False] | np.r_[False, bad[:-1]]
s = np.where(bad, s, np.round(s)); s = np.maximum.accumulate(s)
s = np.where(np.abs(s - np.round(s)) < o.snap, np.round(s), s)
np.save(f"{o.out}_curve.npy", s)
open(f"{o.out}_nuke.txt", "w").write("timingFrame2 {{curve x1 " + " ".join(f"{v + 1:.4g}" for v in s) + "}}\n")
frac = [i for i, v in enumerate(s) if v != round(v)]
st = np.diff(s)
print(f"Deviation take vs plate: max {np.abs(tau - np.arange(len(T))).max():.2f} frames")
print(f"Curve: {len(frac)} in-betweens" + (f" (output index {frac[0]}-{frac[-1]})" if frac else "") + f", steps {st.min():.2f}-{st.max():.2f}")
print(f"Cost median {np.median(cost):.3f}, max {max(cost):.3f} (high = take shows something different from the plate there)")
print("written:", f"{o.out}.npy", f"{o.out}_curve.npy", f"{o.out}_nuke.txt")
