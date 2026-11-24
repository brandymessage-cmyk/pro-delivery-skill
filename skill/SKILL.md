---
name: pro-delivery-skill
description: Turn an AI-generated clip (Seedance, Dreamina, Kling, Veo, Magnific …) into the best possible clean master file: no frame stutter, no freezes, no dropped or duplicate frames, no generator pulse in the background, no wrong first or last frames, correct frame rate, with sound. Measures every frame, repairs with as few invented frames as possible (RIFE), optionally matches the timing of a reference video, and checks the finished file again. Use whenever a generated clip stutters, judders, jumps, flickers, pulses or "breathes", has frame drops, must go from 24p to 25p, must match the timing of an existing clip, or should simply be made as clean as possible before it goes into an edit, a review or a client hand-off, even if the user does not ask for a check explicitly.
---

# Pro Delivery Skill

Generated video often looks fine at a glance and fails as soon as someone scrubs it frame by frame: a frame held and then a double step, a whole generation rhythm, a standing background that refreshes every four frames, a start image that does not move, a hard stop at the end, a file whose timestamps make editing software insert duplicates. This skill finds these errors by measurement, repairs only what is broken, and writes one clean master file with sound. The picture itself (look, colour, content) is not changed.

Scripts are in `scripts/`. Needs `ffmpeg`/`ffprobe`, Python 3 with `numpy` and `opencv-python`, and for repairs RIFE ncnn (`rife-ncnn-vulkan` with model `rife-v4.6`; folder via `RIFE_DIR` or `--rife`). Talk to the user in their language; give concrete frame numbers and factors.

Frame numbers in all scripts are 0-based source frames. A reported transition `i` is the step from frame `i` to `i+1`.

## Rules

1. **Read frame-exact.** Generator downloads often have irregular timestamps (Dreamina: 60 fps time base for 24 frames per second). Every ffmpeg read uses `-fps_mode passthrough`; otherwise ffmpeg silently adds or drops frames. Never load such a file straight into an editing program for the master; build the master with `master.py`.
2. **24p→25p is a re-tag, not a conversion.** Same frames, new clock, 4 % faster, sound with `atempo=25/24`. Never the `fps=25` filter or the editor's frame-rate conversion: they double one frame per second.
3. **Measure at full resolution, step by step.** Mean absolute difference per transition at 960 px width against the direct neighbours (`motion_check.py`). Averages or small previews hide 1.5× jumps.
4. **Repair as little as possible, as wide as necessary.** A frozen frame becomes the midpoint of its real neighbours; a dropped frame is inserted; a double step is spread over about seven transitions, not three. Real frames stay untouched. Frame count and sound stay in sync.
5. **Rhythm, not only single jumps.** A generator cadence (hold + double step every few frames) shows as autocorrelation of the steps even when no single step crosses 1.2×. Fix it with a whole-clip retime onto a smooth time curve (`framefix.py --retime`), not frame by frame.
6. **Locked camera: look for the background pulse.** Generators refresh the standing image every few frames (often 4). People describe it as "the image jumps" or "the picture breathes". `bg_pulse.py --measure` finds it (autocorrelation of background steps, peaks every 4th transition). `bg_pulse.py --out` removes it: temporal trimmed mean over 13 frames plus Gaussian for the standing image, everything that moves stays original. Measure again afterwards.
7. **Do not ship the start image as frame 0.** A generation from a start image usually returns that still as frame 0; motion starts one frame later or a detail swaps. Generate one frame more than needed and start the master at source frame 1 (`master.py` does this by default). Look at the first three frames side by side.
8. **Check the end.** Generators often stop motion abruptly and hold the last frames, which reads as a freeze. Cut before the hold, or ease out.
9. **When a reference video defines the timing, use it as the clock.** If the clip re-creates an existing shot (a plate, an animatic, a cut), `plate_clock.py` measures for every generated frame which reference frame it shows and builds a time curve. A constant offset needs no new frames; in-betweens only where the clip runs slower or faster. `retime_curve.py` applies the curve.
10. **Pick the interpolation by motion.** RIFE handles limbs, hair and fur well. With a locked camera use `--protect`: RIFE only on moving parts, background from the real neighbours, so straight lines behind a moving object do not bend. With a moving camera use the full RIFE frame.
11. **Look at every object at 100 % through the whole clip.** Generators morph details mid-shot (a second needle on a record, a sign that changes, an extra hand) and reproduce details from references that should not be there (a lit brake light). No step measurement finds this. Mark it for the user: fix by a new generation, or by a patch from a clean frame of the same clip.
12. **Smooth a driving video before generating.** Video-to-video and depth-guided models copy the rhythm of their reference. Run `smooth_reference.py` on an old or juddery driving clip first and check it with `motion_check.py`.
13. **Do not "improve" the picture.** No grading, no sharpening, no denoise unless asked. Interpolators and upscalers that sharpen make detail "breathe" from frame to frame (`quality_checks.py` reports it).
14. **The master is checked like the take.** Run the checks on the finished file, not only on the intermediate steps. Frame count and sound length must match.

## Workflow

### 1. Check the take

```bash
python scripts/motion_check.py take.mp4
python scripts/quality_checks.py take.mp4 --png checks/
python scripts/bg_pulse.py take.mp4 --measure          # only with a locked camera
```

- `motion_check.py`: resolution, bit depth, fps, frame count, irregular timestamps; jumps (>1.2× neighbours) and freezes (<0.5×).
- `quality_checks.py`: start image as frame 0, frozen end, clipped highlights, 8-bit banding risk, generator rhythm, sharpness flicker. Look at the images in `checks/`.
- Then play the clip frame by frame at 100 % for morphs and wrong details (rule 11).

### 2. Decide

| Finding | Cause | Action |
|---|---|---|
| Freeze next to a double step | generator holds a frame | `framefix.py` (replaces the held frame) |
| Isolated 1.6–3.6× step | dropped frame | `framefix.py` (inserts, keeps length) |
| Single 1.5–2× step at the same place as in the driving video | inherited stutter | `framefix.py`; next time smooth the reference |
| Rhythm (autocorrelation), no single big step | generator cadence | `framefix.py --retime` |
| Background steps peak every 4th transition, camera locked | background refresh | `bg_pulse.py --out` (rule 6) |
| Clip slower or faster than a reference in one stretch | generation timing | `plate_clock.py` + `retime_curve.py` (rule 9) |
| Small "jump" in a nearly still phase (blink, twitch) | real motion | leave it |
| Step larger only because motion slows down | natural deceleration | leave it |
| Step above 3.6× | a cut | leave it |
| Object morphs, wrong detail | generator | tell the user: regenerate or patch |

### 3. Repair

Frame errors and rhythm, straight to a file with sound:

```bash
python scripts/framefix.py take.mp4 --dry                    # plan only
python scripts/framefix.py take.mp4 fixed.mov [--retime] [--conform 25]
```

It rebuilds missing, held and duplicate frames with RIFE, matches brightness and colour of each new frame to its neighbours, keeps real frames bit-identical and writes ProRes 422 HQ with sound and a framelog. `--length keep` (default) keeps frame count and sound in sync.

Background pulse and timing against a reference (locked camera):

```bash
python scripts/bg_pulse.py take.mp4 --out stab/ --grain 0          # 16-bit PNG per frame
python scripts/bg_pulse.py stab/ --measure                          # must say: no clear pulse
python scripts/plate_clock.py take.mp4 reference.mov --frames 119 --out tau
python scripts/retime_curve.py stab/ tau_curve.npy retimed/ --protect
```

`--grain 0` when the take is almost grainless (typical for Seedance); `auto` puts back what the averaging removed. `plate_clock.py` prints how many in-betweens the curve needs and where; a high cost means the clip shows something different from the reference there.

Order when several apply: background pulse first, then timing (retime), then the master. A clip with frame errors and no reference: `framefix.py` alone.

### 4. Write the master

```bash
python scripts/master.py retimed/ master.mov --audio take.mp4 --fps 25 --src-fps 24 --start 0
python scripts/master.py take.mp4 master.mov --audio take.mp4 --fps 25 --src-fps 24      # no repair needed
```

ProRes 4444 by default (`--codec 422hq` or `--codec h264 --bitrate 40M` for a viewing copy). `--start 1` (default for a video input) skips the start image. Sound starts at the same point, follows a re-tag with `atempo` and is trimmed to the picture. For a viewing copy with a size limit: bitrate = 8 × megabytes / seconds, minus about 5 % for sound.

### 5. Check the master

```bash
python scripts/motion_check.py master.mov
python scripts/quality_checks.py master.mov
python scripts/bg_pulse.py master.mov --measure       # if the pulse was removed
```

Every remaining hint is judged and named: real motion, or an error that needs another round.

## Report to the user

- the file: path, frames, fps, codec, sound
- what was found and what was repaired (frame numbers, number of in-betweens, pulse before/after)
- remaining hints and why they are not errors
- what only the eye can judge (morphs, wrong details) and what you saw

## Measured examples

- Locked shot, van at a gate (Seedance 2.0 4K, 121 frames): background pulse every 4 frames, autocorrelation +0.40 (take 1) and +0.20 (take 2), peaks at 16, 20, 24 …. After `bg_pulse.py`: no pulse. Against the original plate the van ran slow at 47–58 and hurried at 69–72; `plate_clock.py` gave 28 in-betweens at 46–73, everything else real frames, within 0.27 frames of a hand-made Nuke Kronos curve.
- Frame test on a Seedance clip with two built-in drops, a hold and a duplicate: `framefix.py` found all four; rebuilt frames matched the true originals at 41–43 dB (a copied neighbour: 40 dB).
- Detail: the same prompt gave far less fine detail with Seedance 2.5 than with 2.0; a "4K" label can be an upscale. Measure energy above HD or compare 100 % crops before blaming the pipeline.

## Known traps

- `-fps_mode passthrough` missing → silent duplicates
- `fps=25` filter or editor conversion → one duplicate per second
- measuring with brightness averages or previews → missed jumps
- double step spread over three frames → speed bump
- RIFE time `t` is not linear in motion → measure after repair
- background refresh below the jump threshold → only autocorrelation finds it
- protected background taken twice from the same real frame → counts as a freeze; `retime_curve.py` cross-fades the two neighbours
- start image as frame 0, hard stop at the end → jump at the start, freeze at the end
- reference video beats text and start image → details from the reference appear in the generation
- sharpening interpolator or upscaler → detail breathes
- the number in a Dreamina download name is a download counter, not the job; note the generation mode when you start it
