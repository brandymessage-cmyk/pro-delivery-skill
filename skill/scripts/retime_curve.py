"""Retime a clip along a time curve (e.g. from plate_clock.py): output frame i shows source position curve[i].

python retime_curve.py <video|png_folder> curve.npy <out_folder> [--protect] [--thr 6] [--rife DIR]

- Integer curve value: the real source frame, unchanged.
- Fractional value: RIFE v4.6 between the two neighbouring frames at t.
- --protect (locked camera only): RIFE only where something moves; the rest is a dissolve of the two real
  neighbours. That way RIFE bends no straight lines in the background, and the background is never identical
  twice (otherwise the measurement counts a freeze). Best after bg_pulse.py, then the background is already calm.
Output: o_0000.png ... (16 bit) plus framelog.txt. Then master.py for the finished file with sound."""
import os, sys, glob, argparse, subprocess, tempfile, numpy as np, cv2

a = argparse.ArgumentParser(); a.add_argument("src"); a.add_argument("curve"); a.add_argument("out")
a.add_argument("--protect", action="store_true"); a.add_argument("--thr", type=float, default=6)
a.add_argument("--rife", default=os.environ.get("RIFE_DIR", os.path.expanduser("~/SeedanceAgent/tools/rife/rife-ncnn-vulkan-20221029-macos")))
o = a.parse_args(); os.makedirs(o.out, exist_ok=True); S = np.load(o.curve)
tmpd = tempfile.mkdtemp()
if os.path.isdir(o.src):
    files = sorted(glob.glob(os.path.join(o.src, "*.png")) + glob.glob(os.path.join(o.src, "*.tif*")))
else:   # unpack video frame-accurately (passthrough) as 16-bit PNG
    subprocess.run(["ffmpeg", "-v", "error", "-i", o.src, "-fps_mode", "passthrough", "-pix_fmt", "rgb48be", f"{tmpd}/src_%05d.png"], check=True)
    files = sorted(glob.glob(f"{tmpd}/src_*.png"))
print(f"{len(files)} source frames, {len(S)} output frames")
def rd(k):
    x = cv2.imread(files[k], cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32)
    return x / (65535 if x.max() > 255 else 255)
def wr(x, p): cv2.imwrite(p, (np.clip(x, 0, 1)[..., ::-1] * 65535 + 0.5).astype(np.uint16))
Y = lambda x: x @ np.array([.2126, .7152, .0722], np.float32)
rng = np.random.default_rng(0); log = []
for i, s in enumerate(map(float, S)):
    p = f"{o.out}/o_{i:04d}.png"
    if abs(s - round(s)) < 1e-6:
        wr(rd(int(round(s))), p); log.append(f"{i}\treal\tsource {int(round(s))}"); continue
    a0 = int(np.floor(s)); b0 = min(a0 + 1, len(files) - 1); t = round(s - a0, 4); A, B = rd(a0), rd(b0)
    for x, n in ((A, "a"), (B, "b")): cv2.imwrite(f"{tmpd}/{n}.png", (np.clip(x, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))
    subprocess.run([f"{o.rife}/rife-ncnn-vulkan", "-0", f"{tmpd}/a.png", "-1", f"{tmpd}/b.png", "-o", f"{tmpd}/m.png", "-s", str(t),
                    "-m", f"{o.rife}/rife-v4.6", "-z"], check=True, capture_output=True)
    M = cv2.imread(f"{tmpd}/m.png")[..., ::-1].astype(np.float32)
    M = (M + rng.uniform(-0.5, 0.5, M.shape).astype(np.float32)) / 255          # dither against RIFE 8-bit banding steps
    if o.protect:
        near = A * (1 - t) + B * t                     # background: dissolve of the real neighbours (locked camera, no bending)
        d = cv2.GaussianBlur(np.abs(Y(M) - Y(near)), (0, 0), 2.0); d2 = cv2.GaussianBlur(np.abs(Y(A) - Y(B)), (0, 0), 2.0)
        m = cv2.dilate(((d > o.thr / 255) | (d2 > o.thr / 255)).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), 5.0)[..., None]; M = M * m + near * (1 - m)
        log.append(f"{i}\tRIFE\tsource {a0}-{b0} t={t} (moving parts only, {m.mean() * 100:.1f} %)")
    else:
        log.append(f"{i}\tRIFE\tsource {a0}-{b0} t={t}")
    wr(M, p); print(log[-1], flush=True)
open(f"{o.out}/framelog.txt", "w").write("\n".join(log) + "\n")
for f in glob.glob(f"{tmpd}/*"): os.remove(f)
os.rmdir(tmpd)
print("done:", len(S), "frames,", sum("RIFE" in l for l in log), "in-betweens")
