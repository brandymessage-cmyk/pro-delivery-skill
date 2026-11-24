"""Finished master file with sound from an image folder (PNG/TIFF) or a video.

python master.py <png_folder|video> out.mov --audio take.mp4 [--fps 25] [--src-fps 24] [--start 1] [--count N]
                 [--codec 4444|422hq|h264] [--bitrate 40M]

- Reads videos frame-accurately (-fps_mode passthrough); images are taken in name order.
- --start/--count: range in source frames (0-based). Default --start 1: with start-frame generations, frame 0 is
  often still the still image.
- --fps 25 on 24p material = relabel: the same frames, 4 % faster. Never the fps filter (one duplicate frame per second).
  The sound follows with atempo (fps/src-fps) and is trimmed to the picture length.
- Codec: 4444 = ProRes 4444 12 bit (master), 422hq = ProRes 422 HQ, h264 = review file (with --bitrate).
At the end: frame count and sound length are checked and printed."""
import os, sys, glob, json, argparse, subprocess, tempfile

a = argparse.ArgumentParser(); a.add_argument("src"); a.add_argument("out"); a.add_argument("--audio")
a.add_argument("--fps", type=float, default=25); a.add_argument("--src-fps", type=float)
a.add_argument("--start", type=int, default=1); a.add_argument("--count", type=int)
a.add_argument("--codec", choices=["4444", "422hq", "h264"], default="4444"); a.add_argument("--bitrate", default="40M")
o = a.parse_args(); src_fps = o.src_fps or o.fps
tmp = tempfile.mkdtemp()
if os.path.isdir(o.src):
    fs = sorted(glob.glob(os.path.join(o.src, "*.png")) + glob.glob(os.path.join(o.src, "*.tif*")))
else:
    subprocess.run(["ffmpeg", "-v", "error", "-i", o.src, "-fps_mode", "passthrough", "-pix_fmt", "rgb48be", f"{tmp}/f_%05d.png"], check=True)
    fs = sorted(glob.glob(f"{tmp}/f_*.png"))
fs = fs[o.start:] if o.count is None else fs[o.start:o.start + o.count]
if not fs: sys.exit("no frames in the selected range")
ext = os.path.splitext(fs[0])[1]
for k, f in enumerate(fs): os.symlink(os.path.abspath(f), f"{tmp}/s_{k:05d}{ext}")
n = len(fs); dur = n / o.fps
cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", f"{o.fps:g}", "-i", f"{tmp}/s_%05d{ext}"]
if o.audio:
    t0 = o.start / src_fps                                    # sound from the same point as the picture
    cmd += ["-ss", f"{t0:.4f}", "-i", o.audio, "-map", "0:v", "-map", "1:a?"]
    af = f"atempo={o.fps / src_fps:.6f}," if abs(o.fps - src_fps) > 1e-6 else ""
    cmd += ["-af", f"{af}atrim=0:{dur:.4f},apad=whole_dur={dur:.4f}", "-c:a", "pcm_s24le" if o.codec != "h264" else "aac", "-b:a", "320k"]
if o.codec == "4444": cmd += ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuv444p10le", "-vendor", "apl0"]
elif o.codec == "422hq": cmd += ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0"]
else: cmd += ["-c:v", "libx264", "-b:v", o.bitrate, "-maxrate", o.bitrate, "-bufsize", "2M", "-pix_fmt", "yuv420p", "-preset", "slow"]
cmd += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-frames:v", str(n), o.out]
subprocess.run(cmd, check=True)
for f in glob.glob(f"{tmp}/*"): os.remove(f)
os.rmdir(tmp)
j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=codec_type,nb_read_frames,duration",
                               "-of", "json", o.out], capture_output=True, text=True).stdout)["streams"]
v = next(s for s in j if s["codec_type"] == "video"); au = [s for s in j if s["codec_type"] == "audio"]
print(f"{o.out}: {v['nb_read_frames']} frames ({n} expected), {o.fps:g} fps, " +
      (f"sound {float(au[0]['duration']):.3f} s (picture {dur:.3f} s)" if au else "NO sound"))
