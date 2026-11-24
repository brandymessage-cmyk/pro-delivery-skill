"""Shared helpers: read video frame-accurately (no duplicate frames), exact luma from Y codes, read EXR."""
import json,subprocess
import numpy as np

def probe(path):
    """Width, height, bit depth, frame count, whether the timestamps are irregular (VFR)."""
    j=json.loads(subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-count_frames","-show_entries",
        "stream=width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,color_range","-of","json",path],
        capture_output=True,text=True,check=True).stdout)["streams"][0]
    pts=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","frame=pts","-of","csv=p=0",path],
        capture_output=True,text=True).stdout.split()
    d=np.diff([int(x.strip(",")) for x in pts if x.strip(",").lstrip("-").isdigit()])
    j["vfr"]=bool(len(d) and d.min()!=d.max())
    import re as _re; _m=_re.search(r"p(\d+)(le|be)?$",j["pix_fmt"]); j["bits"]=int(_m.group(1)) if _m else 8
    return j

def luma_frames(path,width=960,start=0,count=None):
    """Luma 0..255 per frame, exactly from the Y codes (TV range), with -fps_mode passthrough.
    Without passthrough ffmpeg silently inserts duplicate frames when timestamps are irregular."""
    p=probe(path); w,h=p["width"],p["height"]
    raw=subprocess.run(["ffmpeg","-v","error","-i",path,"-fps_mode","passthrough","-f","rawvideo","-pix_fmt","yuv420p10le","-"],
        capture_output=True,check=True).stdout
    fs=w*h*3//2*2; n=len(raw)//fs; out=[]
    f=max(1,w//width)
    for i in range(start, n if count is None else min(n,start+count)):
        y=np.frombuffer(raw,np.uint16,count=w*h,offset=i*fs).reshape(h,w).astype(np.float32)
        y=(y-64)/876*255
        out.append(y[:h//f*f,:w//f*f].reshape(h//f,f,w//f,f).mean((1,3)))
    return np.array(out)

def exr_rgb(path):
    import OpenEXR,Imath
    e=OpenEXR.InputFile(path); dw=e.header()["dataWindow"]; w,h=dw.max.x+1,dw.max.y+1
    return np.stack([np.frombuffer(e.channel(c,Imath.PixelType(Imath.PixelType.HALF)),np.float16).reshape(h,w) for c in "RGB"],-1).astype(np.float32)

def exr_luma(path,width=960):
    a=exr_rgb(path); h,w=a.shape[:2]; f=w//width
    y=(a@np.array([0.2126,0.7152,0.0722],np.float32))*255
    return y[:h//f*f,:w//f*f].reshape(h//f,f,w//f,f).mean((1,3))

def steps(L):
    """Mean absolute difference per transition (frame i -> i+1)."""
    return np.array([np.abs(L[i+1]-L[i]).mean() for i in range(len(L)-1)])

def flag(d,jump=1.2,freeze=0.5,floor=0.8):
    """Jumps and freezes relative to the direct neighbours. Index = transition i -> i+1."""
    J,F=[],[]
    for i in range(1,len(d)-1):
        nb=(d[i-1]+d[i+1])/2
        if d[i]>jump*nb and d[i]>floor: J.append((i,round(float(d[i]/nb),2)))
        if nb>floor and d[i]<freeze*nb: F.append((i,round(float(d[i]/nb),2)))
    return J,F

def _up(c):
    c=np.repeat(np.repeat(c,2,0),2,1); c[:,1:-1]=0.5*(c[:,:-2]+c[:,2:]); c[1:-1,:]=0.5*(c[:-2,:]+c[2:,:]); return c

def yuv10_to_rgb(b,w,h):
    """yuv420p10le TV range -> RGB float Rec.709 (display-referred), identical to to_exr.py."""
    q=w*h//4; y=b[:w*h].reshape(h,w).astype(np.float32)
    u=_up(b[w*h:w*h+q].reshape(h//2,w//2).astype(np.float32)); v=_up(b[w*h+q:].reshape(h//2,w//2).astype(np.float32))
    Y,Cb,Cr=(y-64)/876,(u-512)/896,(v-512)/896; R,B=Y+1.5748*Cr,Y+1.8556*Cb
    return np.stack([R,(Y-0.2126*R-0.0722*B)/0.7152,B],-1)

def decode_frames(path,indices):
    """Only the requested source frames (0-based) as RGB float, frame-accurate with -fps_mode passthrough."""
    p=probe(path); w,h=p["width"],p["height"]; want=set(indices); out={}
    proc=subprocess.Popen(["ffmpeg","-v","error","-i",path,"-fps_mode","passthrough","-f","rawvideo","-pix_fmt","yuv420p10le","-"],stdout=subprocess.PIPE)
    size=w*h*3//2*2; i=0
    while want-out.keys():
        b=proc.stdout.read(size)
        if len(b)<size: break
        if i in want: out[i]=yuv10_to_rgb(np.frombuffer(b,np.uint16),w,h)
        i+=1
    proc.kill()
    if want-out.keys(): raise SystemExit(f"Source has only {i} frames, missing {sorted(want-out.keys())}")
    return out,w,h
