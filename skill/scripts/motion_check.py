"""Checks a video or an EXR sequence transition by transition for freezes and motion jumps.
Usage: motion_check.py <video.mp4|exr_folder> [--first 1001] [--jump 1.2]
--first = number of the first frame; the target frame of a transition is reported (transition i -> i+1 = first+i+1).
Measures at 960 px width with mean absolute difference. Do not measure with YAVG or a coarse grid:
jumps of 1.5x get lost that way."""
import argparse,glob,os,sys
import numpy as np
sys.path.insert(0,os.path.dirname(__file__)); from _common import *
ap=argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("--first",type=int,default=0); ap.add_argument("--jump",type=float,default=1.2)
a=ap.parse_args()
if os.path.isdir(a.src):
    fs=sorted(glob.glob(os.path.join(a.src,"*.exr"))); L=np.array([exr_luma(f) for f in fs]); print(f"{len(fs)} EXR frames")
else:
    p=probe(a.src); print(f"{p['width']}x{p['height']} {p['pix_fmt']} {p['r_frame_rate']} | {p['nb_read_frames']} frames | VFR timestamps: {'YES, always -fps_mode passthrough' if p['vfr'] else 'no'}")
    L=luma_frames(a.src)
d=steps(L); J,F=flag(d,jump=a.jump)
fn=lambda i:a.first+i+1
print("Steps:"," ".join(f"{fn(i)}:{x:.2f}" for i,x in enumerate(d)))
print("Jumps (target frame, factor):",[(fn(i),r) for i,r in J] or "none")
print("Freezes (target frame, factor):",[(fn(i),r) for i,r in F] or "none")
