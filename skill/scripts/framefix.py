"""FrameFix local: finds missing, duplicate and too-early frames in a generated clip,
rebuilds them with RIFE, matches the generated frames in brightness and colour to their neighbours
and writes ProRes 422 HQ with sound and a framelog. The real frames stay bit-identical (Y codes).

Usage: framefix.py <in.mp4|mov> [out.mov] [--dry] [--length keep|free] [--fps 24] [--conform 25]
       [--insert 40,80] [--replace 12] [--remove 77] [--window 24] [--no-color]

What is detected (index = source frame, 0-based):
  Dropped   transition i->i+1 is 1.6-3.6x as large as its neighbours, with no freeze next to it: Seedance has
            skipped one (from 2.6x: two) frame(s). -> insert between i and i+1 (RIFE t=1/2 or 1/3, 2/3).
  Hold      freeze directly next to a double step (Magnific/Dreamina cadence): one frame sits at the
            wrong time. -> replace with the midpoint of its neighbours, frame count stays.
  Duplicate single freeze with no jump next to it. -> with --length keep removed as compensation for an insert,
            otherwise replaced by the midpoint. With --length free removed.
  Cut       jump above 3.6x: left untouched, only reported.
Multi-frame stills (intended freezes) are left untouched.

--length keep (default): for each inserted frame one frame nearby (±window frames) is removed,
  preferably a duplicate, otherwise the calmest one. Frame count and sound stay in sync; between insert and
  removal the picture shifts by at most one frame against the sound.
--length free: dropped frames are inserted, duplicates removed, the length changes (like BytePlus without
  align_source_fps). Sound stays unchanged, so it lags by one frame after each insert.
--conform 25: afterwards relabel 24p->25p (same frames, 4 % faster, sound atempo 25/24).
Afterwards the script measures the result again and reports remaining jumps."""
import argparse,os,subprocess,sys,tempfile
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from _common import probe,luma_frames,steps,yuv10_to_rgb

ap=argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out",nargs="?")
ap.add_argument("--dry",action="store_true"); ap.add_argument("--length",choices=["keep","free"],default="keep")
ap.add_argument("--fps",type=float); ap.add_argument("--conform",type=float)
ap.add_argument("--insert",default=""); ap.add_argument("--replace",default=""); ap.add_argument("--remove",default="")
ap.add_argument("--window",type=int,default=24); ap.add_argument("--retime",action="store_true"); ap.add_argument("--sigma",type=float,default=1.5); ap.add_argument("--tune",type=int,default=0)   # experimental: worse on a test clip (roughness 11 -> 14 %), MAD share is not a linear motion measure
ap.add_argument("--no-color",action="store_true")
ap.add_argument("--drop",type=float,default=1.6); ap.add_argument("--cut",type=float,default=3.6)
ap.add_argument("--rife",default=os.environ.get("RIFE_DIR",os.path.expanduser("~/SeedanceAgent/tools/rife/rife-ncnn-vulkan-20221029-macos")))
a=ap.parse_args()
ints=lambda s:[int(x) for x in s.split(",") if x.strip()]
out=a.out or os.path.splitext(a.src)[0]+"_framefix.mov"; logp=os.path.splitext(out)[0]+"_framelog.txt"

# ---------- Analyse ----------
p=probe(a.src); W,H=p["width"],p["height"]
fr=a.fps or (lambda n,d:n/d)(*map(int,p["avg_frame_rate"].split("/")))
fr=round(fr,3); fr_s=str(int(fr)) if fr==int(fr) else str(fr)
L=luma_frames(a.src,width=min(960,W)); N=len(L); d=steps(L); FLOOR=0.25   # luma units; below this a step counts as standstill
print(f"{os.path.basename(a.src)}: {W}x{H} {p['pix_fmt']} {fr_s} fps, {N} frames, VFR timestamps {'yes' if p['vfr'] else 'no'}")
def nb(i):  # typical step around transition i, excluding i itself
    s=[d[j] for j in (i-2,i-1,i+1,i+2) if 0<=j<len(d)]; s=sorted(s)[:3]   # ignore the largest neighbour (may itself be an error)
    return max(np.median(s),1e-6)
ratio=[d[i]/nb(i) for i in range(len(d))]
isfreeze=lambda i: 0<=i<len(d) and d[i]<0.5*nb(i) and nb(i)>FLOOR
isjump=lambda i: 0<=i<len(d) and ratio[i]>=a.drop and d[i]>FLOOR
replace,insert,remove,cuts,dups,notes={},{},set(),[],[],[]
used=set()
for i in range(1,len(d)-1):
    if not isfreeze(i) or isfreeze(i-1) or isfreeze(i+1): continue   # skip multi-frame stills
    if isjump(i+1) and ratio[i+1]<a.cut:   # f=i+1 lags (hold before the double step)
        f=i+1; replace[f]=(f-1,f+1,0.5); used|={i,i+1}; notes.append(f"Hold: frame {f} = midpoint {f-1}/{f+1} (freeze {d[i]:.2f}, then {ratio[i+1]:.1f}x)")
    elif isjump(i-1) and ratio[i-1]<a.cut:  # f=i came too early (double step, then freeze)
        f=i; replace[f]=(f-1,f+1,0.5); used|={i-1,i}; notes.append(f"Hold: frame {f} = midpoint {f-1}/{f+1} (before {ratio[i-1]:.1f}x, then freeze)")
    else: dups.append(i+1); used.add(i)
# separate generator cadence (e.g. Seedance: slightly larger step every 4 frames) from real dropped frames
strong=[i for i in range(1,len(d)-1) if ratio[i]>=1.5 and d[i]>FLOOR and i not in used and ratio[i]<a.cut]
drop_thr=a.drop; cadence=None
if len(strong)>=4:
    # fixed phase (cadence runs through) or spacings multiples of P (phase jumps after a dropped frame)
    df=[y-x for x,y in zip(strong,strong[1:]) if y-x<=12]
    best=max(((max(sum(i%P==ph for i in strong)/len(strong),(sum(x%P==0 for x in df)/len(df)) if df else 0),-P,ph) for P in range(3,7) for ph in range(P)))
    share,P,ph=best[0],-best[1],best[2]
    if share>=0.6:
        med=float(np.median([ratio[i] for i in strong])); drop_thr=max(a.drop,1.15*med); cadence=(P,share,med)
        notes.append(f"Generator cadence: every {P} frames a {med:.2f}x step ({share:.0%} of the strong jumps match)."
                     +(" -> whole-shot retime (--retime)." if a.retime else f" Not repairable per frame: use --retime. Here a step only counts as dropped frame from {drop_thr:.2f}x."))
spread=[]
for i in range(len(d)):
    if i in used or not (ratio[i]>=drop_thr and d[i]>FLOOR): continue
    if ratio[i]>=a.cut: cuts.append(i); notes.append(f"Cut/large jump {i}->{i+1} ({ratio[i]:.1f}x): left untouched"); continue
    k=1 if ratio[i]<2.6 else 2; insert[i]=[(i,i+1,round(m/(k+1),3)) for m in range(1,k+1)]
weak=[(i,round(float(ratio[i]),2)) for i in range(1,len(d)-1) if 1.2<=ratio[i]<drop_thr and d[i]>FLOOR and i not in used]
# manual overrides take precedence
for i in ints(a.insert): insert[i]=[(i,i+1,0.5)]; notes.append(f"Override: insert between {i} and {i+1}")
for f in ints(a.replace): replace[f]=(f-1,f+1,0.5); notes.append(f"Override: frame {f} = midpoint")
for f in ints(a.remove): remove.add(f); notes.append(f"Override: frame {f} -> remove")

blocked=set()
busy=lambda j: any(abs(j-f)<=1 for f in replace) or j in remove or j in blocked or any(abs(j-i)<=1 or abs(j-(i+1))<=1 for i in insert)
if a.length=="free":
    for i in insert: notes.append(f"Dropped: {len(insert[i])} frame(s) inserted between {i} and {i+1} ({ratio[i]:.1f}x)")
    for f in dups:
        if not busy(f): remove.add(f); blocked.update((f-1,f+1)); notes.append(f"Duplicate: frame {f} removed")
else:
    # Insert only if a frame nearby can be dropped invisibly (duplicate or standstill). Otherwise spread the
    # double step over ~7 transitions (like plan_retime.py): frame count and sound stay, no frame is dropped.
    for i in sorted(list(insert),key=lambda i:-ratio[i]):
        got=[]
        for _ in insert[i]:
            cand=[]
            for j in range(max(2,i-a.window),min(N-2,i+a.window+1)):
                if busy(j) or j in got: continue
                # If frame j is dropped, d[j-1] and d[j] become one step. Compare with the steps next to it (excluding standstills).
                ref=[d[k] for k in (j-3,j-2,j+1,j+2) if 0<=k<len(d) and d[k]>FLOOR]
                r=(d[j-1]+d[j])/max(np.median(ref) if ref else FLOOR,FLOOR)
                cand.append((r+abs(j-i)/(8*a.window),r,j))
            ok=[c for c in cand if c[1]<=(1.25 if c[2] in dups else 1.15)]
            if not ok: break
            got.append(min(ok)[2])
        if len(got)==len(insert[i]):
            for j in got: remove.add(j); blocked.update((j-1,j+1))
            notes.append(f"Dropped: {len(got)} frame(s) inserted between {i} and {i+1} ({ratio[i]:.1f}x), compensation: frame {got} removed"
                         +(" (duplicate)" if all(j in dups for j in got) else " (standstill)"))
        else:
            del insert[i]; spread.append(i); blocked.update(range(i-4,i+6))
            notes.append(f"Dropped {i}->{i+1} ({ratio[i]:.1f}x): no invisible compensation, double step spread over 7 transitions")
    for f in dups:
        if f not in remove and not busy(f): replace[f]=(f-1,f+1,0.5); notes.append(f"Duplicate: frame {f} = midpoint {f-1}/{f+1}")

# time curve for spread jumps (like plan_retime.py, window 3)
dn=d.copy()
if a.retime:   # whole-shot retime: average steps over the cadence length, smooth slightly, keep the sum (start and end stay)
    P=cadence[0] if cadence else 4
    pad=lambda x,n:np.concatenate([np.full(n,x[0]),x,np.full(n,x[-1])])
    m=np.convolve(pad(d,P),np.ones(P)/P,"same")[P:-P]
    r=int(3*a.sigma)+1; kx=np.exp(-0.5*(np.arange(-r,r+1)/a.sigma)**2); kx/=kx.sum()
    dn=np.convolve(pad(m,r),kx,"same")[r:-r]; dn*=d.sum()/dn.sum()
    replace,insert,remove,spread={},{},set(),[]
    notes=[n for n in notes if n.startswith(("Generator","Cut"))]+[f"Whole-shot retime: cadence {P}, smoothing sigma {a.sigma}"]
for j in ([] if a.retime else spread):
    s0,e0=max(1,j-3),min(len(d)-2,j+3); n=e0-s0+1
    ramp=dn[s0-1]+(dn[e0+1]-dn[s0-1])*np.arange(1,n+1)/(n+1); dn[s0:e0+1]=ramp*dn[s0:e0+1].sum()/ramp.sum()   # on dn: overlapping windows keep the sum
pos=np.interp(np.concatenate([[0],np.cumsum(dn)]),np.concatenate([[0],np.cumsum(d)]),np.arange(N))
# output sequence list
ops=[]
for f in range(N):
    if f in remove: continue
    if f in replace: ops.append(("gen",)+replace[f])
    else:
        q=float(pos[f]); b=int(np.floor(q))
        ops.append(("real",int(round(q))) if abs(q-round(q))<0.02 else ("gen",b,b+1,round(q-b,3)))
    for g in insert.get(f,[]): ops.append(("gen",)+g)
# Fine tuning (SKILL.md rule 7): RIFE does not turn t into motion linearly. Measure every generated frame on 960 px
# proxies: actual motion share tau = |M-A| / (|M-A|+|B-M|), then correct t until tau = target.
def tune_ops(ops):
    gi=[k for k,o in enumerate(ops) if o[0]=="gen"]
    if not gi or a.tune<=0: return ops
    pw=min(960,W); ph=int(round(H*pw/W/2))*2
    raw=subprocess.run(["ffmpeg","-v","error","-i",a.src,"-fps_mode","passthrough","-vf",f"scale={pw}:{ph}:flags=area","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True).stdout
    PF=lambda i: np.frombuffer(raw,np.uint8,count=pw*ph*3,offset=i*pw*ph*3).reshape(ph,pw,3)
    tmp=tempfile.mkdtemp(); ops=list(ops); want={k:ops[k][3] for k in gi}; cur=dict(want); worst=0
    for it in range(a.tune):
        worst=0
        for k in gi:
            _,x,y,_t=ops[k]
            for X,n in ((PF(x),"a"),(PF(y),"b")):
                subprocess.run(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{pw}x{ph}","-i","-",f"{tmp}/{n}.png"],input=X.tobytes(),check=True)
            subprocess.run([f"{a.rife}/rife-ncnn-vulkan","-0",f"{tmp}/a.png","-1",f"{tmp}/b.png","-o",f"{tmp}/m.png","-s",str(cur[k]),"-m",f"{a.rife}/rife-v4.6","-z"],check=True,capture_output=True)
            M=np.frombuffer(subprocess.run(["ffmpeg","-v","error","-i",f"{tmp}/m.png","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True).stdout,np.uint8).reshape(ph,pw,3).astype(np.float32)
            A,B=PF(x).astype(np.float32),PF(y).astype(np.float32); da,db=np.abs(M-A).mean(),np.abs(B-M).mean()
            if da+db<1e-3: continue
            tau=da/(da+db); err=want[k]-tau; worst=max(worst,abs(err))
            cur[k]=float(np.clip(cur[k]+0.8*err,0.02,0.98))
        print(f"  Fine tuning round {it+1}: largest deviation target/actual {worst:.3f}")
        if worst<0.02: break
    for k in gi: ops[k]=ops[k][:3]+(round(cur[k],3),)
    return ops
print("\n".join(notes) or "No dropped frames, holds or duplicates found.")
if weak: print(f"Smaller jumps (1.2-{drop_thr:.2f}x, left untouched):",weak)
print(f"Output: {len(ops)} frames (source {N}), {sum(o[0]=='gen' for o in ops)} generated, {len(remove)} removed")
if a.dry: sys.exit(0)
ops=tune_ops(ops)

# ---------- Build ----------
LO,HI=-0.05,1.10; FS=W*H*3//2*2
def rgb_to_yuv10(rgb):
    R,G,B=rgb[...,0],rgb[...,1],rgb[...,2]; Y=0.2126*R+0.7152*G+0.0722*B
    Cb=(B-Y)/1.8556; Cr=(R-Y)/1.5748
    sub=lambda c:c.reshape(H//2,2,W//2,2).mean((1,3))
    y=np.clip(np.rint(Y*876+64),0,1023); u=np.clip(np.rint(sub(Cb)*896+512),0,1023); v=np.clip(np.rint(sub(Cr)*896+512),0,1023)
    return np.concatenate([y.ravel(),u.ravel(),v.ravel()]).astype(np.uint16).tobytes()
def stats(rgb):
    s=rgb[::4,::4].reshape(-1,3); return s.mean(0),s.std(0)
def rife(A,B,t,tmp):
    for X,n in ((A,"a"),(B,"b")):
        o=(np.clip((X-LO)/(HI-LO),0,1)*255+0.5).astype(np.uint8)
        subprocess.run(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-i","-",f"{tmp}/{n}.png"],input=o.tobytes(),check=True)
    subprocess.run([f"{a.rife}/rife-ncnn-vulkan","-0",f"{tmp}/a.png","-1",f"{tmp}/b.png","-o",f"{tmp}/m.png","-s",str(t),"-m",f"{a.rife}/rife-v4.6","-z"],check=True,capture_output=True)
    b=subprocess.run(["ffmpeg","-v","error","-i",f"{tmp}/m.png","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True).stdout
    g=np.frombuffer(b,np.uint8).reshape(H,W,3).astype(np.float32)+np.random.default_rng(0).uniform(-0.5,0.5,(H,W,3)).astype(np.float32)
    return g/255*(HI-LO)+LO
def match(G,A,B,t):
    (mA,sA),(mB,sB),(mG,sG)=stats(A),stats(B),stats(G)
    mt,st=(1-t)*mA+t*mB,(1-t)*sA+t*sB; g=np.clip(st/np.maximum(sG,1e-6),0.9,1.1)
    return (G-mG)*g+mt, float(((mG-mt)@[0.2126,0.7152,0.0722])*255)

dec=subprocess.Popen(["ffmpeg","-v","error","-i",a.src,"-fps_mode","passthrough","-f","rawvideo","-pix_fmt","yuv420p10le","-"],stdout=subprocess.PIPE)
cache={}; nxt=[0]
def frame(i):
    while nxt[0]<=i:
        b=dec.stdout.read(FS)
        if len(b)<FS: raise SystemExit(f"Source ends at frame {nxt[0]}")
        cache[nxt[0]]=b; nxt[0]+=1
    return cache[i]
has_audio=bool(subprocess.run(["ffprobe","-v","error","-select_streams","a","-show_entries","stream=index","-of","csv=p=0",a.src],capture_output=True,text=True).stdout.strip())
vf=[]
af=["-af","apad"]   # pad sound to picture length, -shortest then cuts at the picture end
if a.conform: vf=["-vf",f"setpts=N/{a.conform}/TB"]; af=["-af",f"atempo={a.conform/fr},apad"]
rate=str(a.conform) if a.conform else fr_s
cmd=["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","yuv420p10le","-s",f"{W}x{H}","-r",fr_s,"-i","-"]
if has_audio: cmd+=["-i",a.src,"-map","0:v","-map","1:a:0"]+af+["-c:a","pcm_s24le","-shortest"]
cmd+=vf+["-r",rate,"-c:v","prores_ks","-profile:v","3","-vendor","apl0","-pix_fmt","yuv422p10le",
         "-color_primaries","bt709","-color_trc","bt709","-colorspace","bt709","-color_range","tv",out]
enc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
log=[]
with tempfile.TemporaryDirectory() as tmp:
    for k,o in enumerate(ops):
        if o[0]=="real":
            enc.stdin.write(frame(o[1])); log.append(f"{k}\treal\tsource {o[1]}")
        else:
            _,x,y,t=o; A=yuv10_to_rgb(np.frombuffer(frame(x),np.uint16),W,H); B=yuv10_to_rgb(np.frombuffer(frame(y),np.uint16),W,H)
            G=rife(A,B,t,tmp); dl=0.0
            if not a.no_color: G,dl=match(G,A,B,t)
            enc.stdin.write(rgb_to_yuv10(G)); log.append(f"{k}\tRIFE\tsource {x}-{y} t={t}\tcolour {dl:+.2f}/255 luma corrected")
            print(f"  Frame {k}: RIFE {x}-{y} t={t}, brightness was {dl:+.2f}/255 off before")
        for j in [j for j in cache if j<(o[1] if o[0]=="real" else o[1])-1]: del cache[j]
enc.stdin.close(); enc.wait(); dec.kill()
if enc.returncode: sys.exit("ffmpeg encoder failed")
open(logp,"w").write(f"# {a.src} -> {out}\n# length={a.length} conform={a.conform}\n"+"\n".join(notes)+"\n"+"\n".join(log)+"\n")

# ---------- Re-measure ----------
L2=luma_frames(out,width=min(960,W)); d2=steps(L2)
gen=[k for k,o in enumerate(ops) if o[0]=="gen"]
def r2(i):
    s=sorted([d2[j] for j in (i-2,i-1,i+1,i+2) if 0<=j<len(d2)])[:3]; return d2[i]/max(np.median(s),1e-6)
lim=max(1.2,1.1*cadence[2]) if cadence and not a.retime else 1.2   # generator cadence is already in the source, not a repair error
near=[(i,round(float(r2(i)),2)) for i in range(1,len(d2)-1) if any(abs(i-g)<=1 or abs(i+1-g)<=1 for g in gen) and r2(i)>lim and d2[i]>FLOOR]
print(f"Done: {out} ({len(L2)} frames), framelog {logp}")
print(f"After repair above {lim:.2f}x at generated positions (transition, factor):",near or "none")
if cadence and not a.retime: print(f"HINT: generator cadence every {cadence[0]} frames remains; use --retime for that.")
