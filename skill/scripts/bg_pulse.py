"""Find and remove the background pulse (the generator refreshes the static image every N frames).

Measure:  python bg_pulse.py take.mp4 --measure [--plate plate.mov]   (motion mask from the plate, otherwise from the take)
Fix:      python bg_pulse.py take.mp4 --out stab/ [--thr 4] [--window 13] [--sigma 2] [--grain auto|0]

Locked camera only. Per-pixel trimmed mean over `window` frames (middle ~55 %), then a Gaussian over
time (sigma): single-frame glitches and the pulse disappear, slow lighting changes stay. Where the take deviates from
the smoothed image by more than `thr`/255 (luma, spatially pre-smoothed), something is moving: there the take stays
original (mask dilated, soft). Output: 16-bit PNG per frame (f_0000.png ...) plus mask_stats.txt.
--measure compares step sizes in the static image: autocorrelation at one lag (usually 4) > +0.15 = pulse.
The same measurement on the output folder afterwards (--measure stab/) shows whether it is gone."""
import os, sys, glob, argparse, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import probe

def lumas(src, w=960):
    if os.path.isdir(src):
        fs = sorted(glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.tif*")))
        out = []
        for f in fs:
            x = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32); x /= 65535 if x.max() > 255 else 255
            x = cv2.resize(x, (w, int(round(x.shape[0] * w / x.shape[1]))), interpolation=cv2.INTER_AREA)
            out.append(x[..., ::-1] @ np.array([.2126, .7152, .0722], np.float32))
        return np.array(out)
    p = probe(src); h = int(round(p["height"] * w / p["width"])) // 2 * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-fps_mode", "passthrough", "-vf", f"scale={w}:{h}:flags=area",
                          "-f", "rawvideo", "-pix_fmt", "gray16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint16).reshape(-1, h, w).astype(np.float32) / 65535

def measure(src, plate=None):
    Y = lumas(src); n = len(Y); M = lumas(plate) if plate else Y
    mot = np.abs(np.diff(np.stack([cv2.GaussianBlur(x, (0, 0), 1.2) for x in M]), axis=0)).max(0)
    moving = cv2.dilate((cv2.GaussianBlur(mot, (0, 0), 3) > 0.03).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    bg = ~moving
    if bg.mean() < 0.05: print("  hardly any static image - measurement not meaningful (camera moving?)"); return
    step = np.array([np.abs(Y[i + 1] - Y[i])[bg].mean() for i in range(n - 1)]) * 255
    d = step - np.convolve(step, np.ones(9) / 9, "same"); d = d[4:-4]
    ac = {L: float(np.corrcoef(d[:-L], d[L:])[0, 1]) for L in (2, 3, 4, 5, 6)}
    peaks = [i for i in range(1, n - 2) if step[i] > 1.4 * np.median(step[max(0, i - 3):i + 4])]
    print(f"{src}\n  static image: {bg.mean() * 100:.0f} % | step median {np.median(step):.3f}/255, max {step.max():.3f}")
    print("  autocorrelation " + " ".join(f"Lag{L} {v:+.2f}" for L, v in ac.items()))
    print(f"  peaks (transition i->i+1, >1.4x surroundings): {peaks}")
    L = max(ac, key=ac.get)
    print(f"  => pulse every {L} frames ({ac[L]:+.2f})" if ac[L] > 0.15 else "  => no clear pulse")

def stabilize(src, out, thr, win, sig, grain_mode):
    os.makedirs(out, exist_ok=True); p = probe(src); W, H = p["width"], p["height"]
    raw = None; ST = 270 if H >= 2000 else H; MARG = 32
    k = np.arange(-(win // 2), win // 2 + 1); g = np.exp(-0.5 * (k / sig) ** 2); g /= g.sum()
    keep = max(1, int(round(win * 0.55))); cut = (win - keep) // 2
    rng = np.random.default_rng(7); _t = rng.normal(0, 1, (512, 512)).astype(np.float32)
    HPN = float((_t - cv2.GaussianBlur(_t, (0, 0), 1.5)).std()); grain = None; cover = []
    Yw = np.array([.2126, .7152, .0722], np.float32)
    for y0 in range(0, H, ST):
        a0, a1 = max(0, y0 - MARG), min(H, y0 + ST + MARG)
        pr = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-fps_mode", "passthrough", "-vf", f"crop={W}:{a1 - a0}:0:{a0}",
                             "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"], capture_output=True, check=True).stdout
        T = np.frombuffer(pr, np.uint16).reshape(-1, a1 - a0, W, 3).astype(np.float32) / 65535; N = len(T)
        tm = np.empty_like(T)
        for i in range(N):
            lo = min(max(0, i - win // 2), max(0, N - win)); s = np.sort(T[lo:lo + win], 0); tm[i] = s[cut:cut + keep].mean(0)
        S = np.empty_like(T)
        for i in range(N): S[i] = np.tensordot(g, tm[np.clip(i + k, 0, N - 1)], axes=(0, 0))
        if grain is None:
            if grain_mode == "auto":
                hp = lambda X: np.stack([x - cv2.GaussianBlur(x, (0, 0), 1.5) for x in X[::10]]).var((0, 1, 2))
                grain = np.sqrt(np.maximum(hp(T) - hp(S), 0)); print("grain compensation per channel (std):", grain.round(5))
            else:
                grain = np.full(3, float(grain_mode), np.float32)
        lo = y0 - a0; hgt = min(ST, H - y0)
        for i in range(N):
            d = cv2.GaussianBlur(np.abs((T[i] - S[i]) @ Yw), (0, 0), 2.0)
            m = cv2.dilate((d > thr / 255).astype(np.uint8), np.ones((13, 13), np.uint8)).astype(np.float32)
            m = cv2.GaussianBlur(m, (0, 0), 4.0)[..., None]
            o = S[i]
            if grain.max() > 0:
                n = rng.normal(0, 1, T[i].shape).astype(np.float32); o = o + (n - cv2.GaussianBlur(n, (0, 0), 1.5)) * (grain / HPN)
            o = T[i] * m + o * (1 - m)
            np.save(f"{out}/s{y0:05d}_{i:04d}.npy", np.clip(o[lo:lo + hgt], 0, 1).astype(np.float16)); cover.append((i, float(m[lo:lo + hgt].mean())))
        print("strip", y0, "done", flush=True)
    strips = sorted({f.split("/")[-1][1:6] for f in glob.glob(f"{out}/s*_0000.npy")})
    for i in range(N):
        f = np.concatenate([np.load(f"{out}/s{s}_{i:04d}.npy") for s in strips], 0).astype(np.float32)
        cv2.imwrite(f"{out}/f_{i:04d}.png", (f[..., ::-1] * 65535 + 0.5).astype(np.uint16))
        for s in strips: os.remove(f"{out}/s{s}_{i:04d}.npy")
    c = np.zeros(N)
    for i, v in cover: c[i] += v / len(strips)
    open(f"{out}/mask_stats.txt", "w").write("\n".join(f"{i}\t{c[i] * 100:.1f} % original (moving)" for i in range(N)) + "\n")
    print(f"done: {N} frames, take share (moving) mean {c.mean() * 100:.1f} %, max {c.max() * 100:.1f} %")

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("src"); a.add_argument("--measure", action="store_true"); a.add_argument("--plate"); a.add_argument("--out")
    a.add_argument("--thr", type=float, default=4); a.add_argument("--window", type=int, default=13)
    a.add_argument("--sigma", type=float, default=2.0); a.add_argument("--grain", default="auto")
    o = a.parse_args()
    if o.measure or not o.out: measure(o.src, o.plate)
    else: stabilize(o.src, o.out, o.thr, o.window, o.sigma, o.grain)
