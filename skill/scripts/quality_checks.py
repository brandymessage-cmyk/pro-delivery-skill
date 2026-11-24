"""Additional checks for generated clips (video or image folder). Extends motion_check.py with
errors that post houses have flagged and that a pure jump/freeze search misses:

  1. First frame = reference image: change 0->1 per image region against the following changes.
  2. Frozen end: motion stops abruptly in the last frames.
  3. Clipping: share of flat plateaus >= 1.0 (blown out, no structure) and <= 0. Reconstructed highlights above 1.0 with structure are allowed.
  4. Banding: report 8-bit source, check image of the smoothest area at 6x contrast (--png).
  5. Generator cadence: recurring rhythm of step sizes (hold + double step), autocorrelation.
  6. Sharpness flicker: detail energy fluctuates from frame to frame (typical after AI interpolation/upscaling).

Usage: quality_checks.py <video> [--first 0] [--png <folder>]
Output: OK or HINT with values per check. Exit 0 always (hints are judged by a human), 2 on read error.
--png writes check images: clipping in red, the first three frames side by side, smoothest area at 6x contrast."""
import argparse,glob,os,subprocess,sys
import numpy as np
sys.path.insert(0,os.path.dirname(__file__)); from _common import probe,exr_rgb

ap=argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("--first",type=int,default=0); ap.add_argument("--png")
a=ap.parse_args()
W=1920   # working width for motion and sharpness

def frames():
    """Yields RGB float per frame (working width) and the original size. Video always with -fps_mode passthrough."""
    if os.path.isdir(a.src):
        for f in sorted(glob.glob(os.path.join(a.src,"*.exr"))):
            x=exr_rgb(f); h,w=x.shape[:2]; yield x,(w,h)
        return
    p=probe(a.src); w,h=p["width"],p["height"]; hh=int(round(h*W/w/2))*2
    proc=subprocess.Popen(["ffmpeg","-v","error","-i",a.src,"-fps_mode","passthrough","-vf",f"scale={W}:{hh}:flags=area",
        "-f","rawvideo","-pix_fmt","rgb48le","-"],stdout=subprocess.PIPE)
    n=W*hh*3*2
    while True:
        b=proc.stdout.read(n)
        if len(b)<n: break
        yield np.frombuffer(b,np.uint16).reshape(hh,W,3).astype(np.float32)/65535,(w,h)
    proc.kill()

def plateau(x):
    """Blown out = values >= 1.0 AND flat (5x5 neighbourhood varies < 0.01). Reconstructed highlights above 1.0 with structure do not count."""
    import cv2
    mx=x.max(-1); hi=mx>=0.999
    if not hi.any(): return 0.0,0.0
    k=np.ones((5,5),np.uint8); rng=cv2.dilate(mx,k)-cv2.erode(mx,k)
    return float((hi&(rng<0.01)).mean()), float(hi.mean())

def small(x,f):
    h,w=x.shape[:2]; return x[:h//f*f,:w//f*f].reshape(h//f,f,w//f,f).mean((1,3))

L,HF,CLIPW,CLIPB,keep,size=[],[],[],[],[],None
for i,(x,size) in enumerate(frames()):
    if x.shape[1]>W:   # EXR at full resolution: measure clipping here, the rest at working width
        pw,_=plateau(x); CLIPW.append(pw); CLIPB.append(float((x<=0.0).all(-1).mean()))
        f=x.shape[1]//W; x=x[:x.shape[0]//f*f,:x.shape[1]//f*f].reshape(x.shape[0]//f,f,x.shape[1]//f,f,3).mean((1,3))
    else:
        pw,_=plateau(x); CLIPW.append(pw); CLIPB.append(float((x<=0.0).all(-1).mean()))
    y=x@np.array([0.2126,0.7152,0.0722],np.float32)
    L.append(small(y,2)*255)                                     # 960 wide, 0..255
    HF.append(float(np.abs(np.diff(y,axis=1)).mean()))
    if i<7 or i%max(1,10)==0: keep.append((i,x))
if len(L)<8: print("Too few frames read."); sys.exit(2)
L=np.array(L); n=len(L); d=np.array([np.abs(L[i+1]-L[i]).mean() for i in range(n-1)])
print(f"{a.src}: {n} frames, {size[0]}x{size[1]}")
res=[]

# 1. First frame: per tile (8x8) change 0->1 against the median of changes 1->2 .. 5->6
T=8; hh,ww=L.shape[1]//T,L.shape[2]//T
def tiles(k): return np.abs(L[k+1]-L[k])[:hh*T,:ww*T].reshape(T,hh,T,ww).mean((1,3))
first=tiles(0); ref=np.median(np.stack([tiles(k) for k in range(1,min(6,n-1))]),0)
moving=ref>max(1.0,np.percentile(ref,50))
stuck=moving&(first<0.3*ref); frac=float(stuck.sum())/max(1,int(moving.sum()))
msg=f"{frac*100:.0f} % of the moving image regions stand still in the first change"
if frac>=0.15:
    res.append(("HINT","First frame",msg+" -> frame 0 may be the reference/start image. Look at the check image (--png). For start-frame generations always generate one frame more and deliver from source frame 1."))
else: res.append(("OK","First frame",msg+"; the number does not catch every case (local jumps, e.g. reflections) -> look at the check image with --png"))

# 2. Frozen end: last steps against the median of the motion
med=float(np.median(d)); t6=d[-6:]
drops=[float(t6[i+1]/max(t6[i],1e-6)) for i in range(len(t6)-1)]
if med>0.8 and t6[-3:].min()<0.25*med and min(drops[-3:])<0.3:
    k=int(np.argmin(drops[-3:])); res.append(("HINT","End",f"steps at the end {[round(float(x),2) for x in t6]} at median {med:.2f}: motion stops abruptly (drop to {min(drops[-3:])*100:.0f} % from one frame to the next). Build an ease-out or generate longer and cut earlier."))
else: res.append(("OK","End",f"last steps {[round(float(x),2) for x in t6[-4:]]}, median {med:.2f}"+(" (even deceleration)" if t6[-1]<0.5*med else "")))

# 3. Clipping
cw,cb=max(CLIPW)*100,max(CLIPB)*100
s=("HINT" if cw>0.1 or cb>1.0 else "OK")
res.append((s,"Clipping",f"flat white plateaus (>=1.0 without structure) max {cw:.3f} % of pixels, black (<=0) max {cb:.3f} %"+(" -> blown-out/crushed areas cannot be graded; look at the spots (--png), reconstruct highlights or regenerate if needed." if s=="HINT" else "")))

# 4. Banding: 8-bit/4:2:0 sources produce steps and blocks that only show up in grading
if os.path.isdir(a.src): res.append(("INFO","Banding","EXR folder: check the source; look at smooth areas in the check image (--png, contrast 6x)"))
else:
    q=probe(a.src); s8=q["bits"]==8 or "420" in q["pix_fmt"]
    res.append(("HINT" if q["bits"]==8 else "OK","Banding",f"source {q['pix_fmt']} ({q['bits']} bit)"+(" -> 8-bit generator download: smooth areas (sky, walls, defocus) have steps and blocks. Look at the check image; if needed deband in float (smooth areas only), then fine grain. Never on client plates." if q["bits"]==8 else "")))

# 5. Generator cadence: autocorrelation of the step sizes after removing the trend
tr=np.convolve(d,np.ones(9)/9,"same"); r=(d-tr)[4:-4]
def ac(k): return float(np.corrcoef(r[:-k],r[k:])[0,1]) if len(r)>k+4 and r.std()>0 else 0.0
a2,a3,a4=ac(2),ac(3),ac(4); rough=float(np.abs(np.diff(d)).mean()/max(med,1e-6))
s="HINT" if max(a3,a4)>0.25 else "OK"
res.append((s,"Cadence",f"autocorrelation lag 2/3/4: {a2:+.2f} / {a3:+.2f} / {a4:+.2f}, roughness {rough*100:.0f} %"+(" -> recurring generator rhythm (hold + double step). Consider a whole-shot retime to a smooth time curve (Kronos/optical flow for rigid subjects)." if s=="HINT" else "")))

# 6. Sharpness flicker
h=np.array(HF); htr=np.convolve(h,np.ones(9)/9,"same"); fl=float(((h-htr)[4:-4]).std()/max(h.mean(),1e-9))*100
s="HINT" if fl>3.0 else "OK"
res.append((s,"Sharpness",f"fluctuation of detail energy {fl:.1f} % (frame to frame)"+(" -> detail 'breathes' (typical of AI interpolation/upscalers with resharpening)." if s=="HINT" else "")))

for s,k,m in res: print(f"  {s:8s} {k:13s} {m}")

if a.png:
    import cv2; os.makedirs(a.png,exist_ok=True)
    i,x=max(keep,key=lambda t:(t[1]>=0.999).any(-1).mean()); v=np.clip(x,0,1).copy(); v[(x>=0.999).any(-1)]=[1,0,0]
    cv2.imwrite(os.path.join(a.png,f"clipping_{a.first+i}.png"),(v[...,::-1]*255).astype(np.uint8))
    f3=[k[1] for k in keep[:3]]; cv2.imwrite(os.path.join(a.png,f"first_frames_{a.first}-{a.first+2}.png"),(np.clip(np.hstack(f3),0,1)[...,::-1]*255).astype(np.uint8))
    g=keep[min(3,len(keep)-1)][1]; gy=g@np.array([.2126,.7152,.0722],np.float32); B=128
    hs,ws=gy.shape[0]//B,gy.shape[1]//B; bl=gy[:hs*B,:ws*B].reshape(hs,B,ws,B); e=np.abs(np.diff(bl,axis=3)).mean((1,3)); rg=bl.max((1,3))-bl.min((1,3))
    e[rg<0.02]=9; by,bx=np.unravel_index(np.argmin(e),e.shape); y0,x0=max(0,by*B-B),max(0,bx*B-2*B)
    c=g[y0:y0+3*B,x0:x0+5*B]; c=np.clip((c-c.mean())*6+0.5,0,1)
    cv2.imwrite(os.path.join(a.png,"banding_contrast6x.png"),(c[...,::-1]*255).astype(np.uint8))
    print("Images:",a.png)
