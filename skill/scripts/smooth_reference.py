"""Smooths the motion of a reference/driving video BEFORE it becomes a depth map or motion reference.
Seedance adopts the cadence of the reference: stutter in it ends up in the result.
Start and end stay fixed, frame count stays the same; in-between times via RIFE.
Because RIFE times are not linear in picture motion, the script re-measures the result and retunes the times
in --iter rounds, damped with --gain; a round only counts if the action deviates at most --max-drift frames
from the edit (best result counts). Remaining jumps are repaired in the take after generation. --hold-tail N: the last N frames are held frames
(edit longer than source) and are not smoothed but appended as the last real frame.
Usage: smooth_reference.py <in.mp4> <out_dir> [--sigma 2.5] [--iter 3] [--hold-tail 0] [--rife <folder>]
Output: out_dir/o_%04d.png (then encode with -framerate 24, frame for frame to the edit), pos.txt."""
import argparse,glob,os,shutil,subprocess,sys
import numpy as np
from PIL import Image
sys.path.insert(0,os.path.dirname(__file__)); from _common import *
ap=argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--sigma",type=float,default=2.5)
ap.add_argument("--iter",type=int,default=3); ap.add_argument("--gain",type=float,default=0.5); ap.add_argument("--max-drift",type=float,default=1.5); ap.add_argument("--hold-tail",type=int,default=0)
ap.add_argument("--rife",default=os.environ.get("RIFE_DIR",os.path.expanduser("~/SeedanceAgent/tools/rife/rife-ncnn-vulkan-20221029-macos")))
a=ap.parse_args()
shutil.rmtree(a.out,ignore_errors=True); src=os.path.join(a.out,"src"); os.makedirs(src)
subprocess.run(["ffmpeg","-v","error","-y","-i",a.src,"-fps_mode","passthrough","-start_number","0",f"{src}/s_%04d.png"],check=True)
total=len(glob.glob(f"{src}/s_*.png")); n=total-a.hold_tail
def lum(p):
    g=np.asarray(Image.open(p).convert("L"),np.float32); f=max(1,g.shape[1]//960)
    return g[:g.shape[0]//f*f,:g.shape[1]//f*f].reshape(g.shape[0]//f,f,g.shape[1]//f,f).mean((1,3))
d=steps(np.array([lum(f"{src}/s_{i:04d}.png") for i in range(n)]))
x=np.arange(-8,9); k=np.exp(-x**2/(2*a.sigma**2)); k/=k.sum()
v=np.convolve(np.pad(d,8,mode="reflect"),k,"valid")
c=np.concatenate([[0],np.cumsum(v)]); c/=c[-1]; C=np.concatenate([[0],np.cumsum(d)]); C/=C[-1]
def render(pos,tag):
    for i,p in enumerate(pos):
        b=int(np.floor(p)); t=p-b; dst=f"{a.out}/{tag}_{i:04d}.png"
        if t<0.02 or b>=n-1: shutil.copy(f"{src}/s_{min(b,n-1):04d}.png",dst)
        elif t>0.98: shutil.copy(f"{src}/s_{b+1:04d}.png",dst)
        else: subprocess.run([f"{a.rife}/rife-ncnn-vulkan","-0",f"{src}/s_{b:04d}.png","-1",f"{src}/s_{b+1:04d}.png","-o",dst,"-s",f"{t:.4f}","-m",f"{a.rife}/rife-v4.6","-z"],check=True,capture_output=True)
    return steps(np.array([lum(f"{a.out}/{tag}_{i:04d}.png") for i in range(n)]))
def worst(s):
    J,F=flag(s); return max([r for _,r in J]+[1/max(r,1e-3) for _,r in F]+[1.0]),J,F
target=c.copy(); best=None
for it in range(a.iter+1):
    pos=np.interp(target,C,np.arange(n)); pos=np.maximum.accumulate(pos); pos[0],pos[-1]=0,n-1
    s=render(pos,"t"); w,J,F=worst(s); drift=float(np.abs(pos-np.arange(n)).max())
    print(f"Round {it}: worst transition {w:.2f}x | shift {drift:.2f} frames | jumps {J or 'none'} | freezes {F or 'none'}",flush=True)
    if drift<=a.max_drift and (best is None or w<best[0]):
        best=(w,pos.copy())
        for f in glob.glob(f"{a.out}/o_*.png"): os.remove(f)
        for i in range(n): os.replace(f"{a.out}/t_{i:04d}.png",f"{a.out}/o_{i:04d}.png")
    if w<1.15: break
    A=np.concatenate([[0],np.cumsum(s)]); A/=A[-1]
    target=np.clip(target-a.gain*(A-c),0,1); target=np.maximum.accumulate(target); target[0],target[-1]=0,1
for f in glob.glob(f"{a.out}/t_*.png"): os.remove(f)
if best is None: sys.exit(f"No round within --max-drift {a.max_drift} frames; choose a smaller sigma")
for i in range(n,total): shutil.copy(f"{a.out}/o_{n-1:04d}.png",f"{a.out}/o_{i:04d}.png")
pos=np.concatenate([best[1],np.full(a.hold_tail,n-1.0)])
np.savetxt(os.path.join(a.out,"pos.txt"),pos,fmt="%.3f")
print(f"max. shift of the action (frames): {np.abs(best[1]-np.arange(n)).max():.2f} | best result {best[0]:.2f}x | {total} frames ({a.hold_tail} held) -> {a.out}")
