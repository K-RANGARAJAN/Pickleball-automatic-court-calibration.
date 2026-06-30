# Pickleball Court — Data Collection Handoff Spec

**Purpose:** Collect real-world, ground-truthed footage to develop and validate
(1) court calibration → near-half ROI, and (2) ball **point-of-pitch** detection
with two-camera redundancy. Written so a new team can use the data without the
original collector present. **Read this fully before shooting — undocumented
footage is unusable.**

---

## 0. What this data must enable
- Calibrate each camera to the court (keypoints → homography/PnP).
- Detect the ball each frame (train/fine-tune a ball detector).
- Find the **bounce (pitch) point**: at contact the ball is on the court (Z=0),
  so each camera maps it via homography; the two cameras cross-check and cover
  each other when one view is occluded.
- (Future) Triangulate ball height/trajectory from the two synced views.

---

## 1. The rig
- **Camera A & B (main pair):** two cameras, **wide apart**, both behind the near
  baseline (e.g. behind/near each near corner). Synced, 60fps, 1080p.
  - **Each must independently see the FULL near half AND all 12 court keypoints.**
    If a camera loses court lines/corners, it cannot be calibrated → its data is dead.
- **Camera K (kitchen):** single camera at the net post, aimed along the net at the
  non-volley zone. **Separate pipeline** (see §8). Pure viewer.
- All three roll simultaneously for every capture.

### 12-keypoint order (must match the model)
```
0 bg-baseline-L   1 bg-baseline-M   2 bg-baseline-R
3 bg-kitchen-R    4 bg-kitchen-M    5 bg-kitchen-L
6 fg-kitchen-L    7 fg-kitchen-M    8 fg-kitchen-R
9 fg-baseline-R   10 fg-baseline-M  11 fg-baseline-L
(bg = far side, fg = near side)
```

---

## 2. Camera settings — identical on all cameras, locked, never changed mid-session
- **60 fps, constant** (standard video, not Slo-mo). Verify it is NOT variable frame rate.
- **1080p**, same on every camera.
- **Zoom locked at 1.0× (main lens). Never pinch-zoom.** Changing zoom voids calibration.
- **Lock focus AND exposure** (iPhone: tap-and-hold = AE/AF lock).
- **HDR / Dolby Vision OFF** (Settings → Camera → Record Video → HDR Video off).
- **Landscape, orientation locked. Clean the lens.**

---

## 3. One-time calibration captures (do not skip)

### 3a. Intrinsics — per camera
- Print a **checkerboard** (e.g. 9×6 inner corners) on a **rigid flat board**.
  Measure square size in mm precisely → **record it**.
- For **each camera separately**, at the **exact settings/zoom** used for shooting:
  record ~60s slowly moving the board to cover the whole frame — center, all four
  corners, tilted in every direction, near and far. Slow (no blur). ~20+ views.
- Intrinsics are tied to that exact zoom+resolution. Lock before, don't change after.

### 3b. Court reference (extrinsics) — per camera
- Record a few seconds of the **empty near half** from each camera's fixed position,
  all court lines + 12 keypoints clearly visible.
- Lets the team solve each camera's homography/pose.
- **If any camera is bumped/moved, re-shoot its reference.**

### 3c. Measured camera locations
- For each camera, measure and log **position (X, Y from court origin) + height**,
  plus rough aim. (Coarse ground truth; keypoint PnP refines it.)
- **Photograph the full setup.**

---

## 4. Court coordinate system (define once, write it down)
- Measure actual court with a tape: baseline width, sideline length, kitchen depth,
  net height. Confirm regulation (20 × 44 ft, kitchen 7 ft) or note deviations.
- **Origin & axes (must match the model):** e.g. near-baseline-left = (0,0,0),
  X along baseline, Y toward net, Z up. State it explicitly in the README.

---

## 5. Sync (every clip)
- Cameras are synced, but still record a **sharp sync event** at the **start of each
  clip**: a clap/flash visible to A, B (and K) at once. Cheap insurance for alignment
  and for associating the same rally across cameras.
- For pitch-redundancy, loose sync suffices (each camera self-computes the bounce).
  The sync event also preserves the option for tight-sync 3D triangulation later.

---

## 6. What to capture (the data)

### 6a. Ground-truth bounces (the priority)
- Mark known positions on the near half with tape/cones; **record each marker's
  measured court coordinates** in the log.
- Bounce balls onto/near the markers. **Heavy emphasis near the lines** (in/out
  edge cases), kitchen line, baseline, sidelines, corners.
- Many repetitions, captured by A **and** B simultaneously.

### 6b. Deliberate occlusion cases (the redundancy dataset)
- Stage bounces where **a player/paddle blocks ONE camera's view** while the other
  sees it clearly. Block A sometimes, B other times.
- Log which camera was occluded each time.
- Without these, you have two single-camera datasets, not a redundancy dataset.

### 6c. Real rallies
- Varied speeds, spins, angles, rally lengths — for ball-detector training and
  realistic bounce conditions.

### 6d. Variety
- Different lighting (if outdoor: morning / noon / evening — sun angle + shadows).
- Different ball colours if available.

---

## 7. Quantity targets (aim for)
- **Ground-truth bounces:** several hundred, spread across the near half, line-heavy.
- **Occlusion bounces:** ~100+, mixed which-camera-blocked.
- **Ball-visible frames overall:** thousands (for detector training).
- More variety > more of the same scene.

---

## 8. Kitchen camera (Camera K) — separate pipeline
- **Role:** observe close-net / kitchen play the wide pair covers poorly. Pure viewer.
- **Settings:** same locks as §2 (60fps, 1080p, AE/AF lock, no zoom, HDR off).
- **Placement:** net post, aimed along the net across the non-volley zone.
- **Capture:** rolls simultaneously with A & B; same sync event; same clip number.
- **Not** used for calibration/triangulation in this phase — kept as an independent
  stream for later kitchen-specific analysis. Still log it like the others.

---

## 9. Ground-truth protocol (so accuracy can be measured)
- Tape/cone markers at **measured** positions; coordinates in the log.
- For each ground-truth bounce, note the target marker (or that it landed on it).
- Optional but valuable: lay a measuring tape / pre-marked grid for some captures.

---

## 10. File naming & master log

### Filenames
```
YYYY-MM-DD_S<session>_cam<A|B|K>_clip<NNN>.mov
e.g. 2026-07-10_S1_camA_clip007.mov
```
- **Same clip number across cameras for the same moment** (camA_clip007 = camB_clip007 = camK_clip007).

### Master log (spreadsheet, one row per clip) — see DATA_COLLECTION_LOG_TEMPLATE.csv
Columns: filename, date, session, camera, fps, resolution, content_type
(calibration / empty-court / ground-truth-bounce / rally / occlusion),
sync_event (y/n), occluded_camera (if any), gt_markers_present (y/n), notes.

### README (ship alongside footage) — must capture:
- Court origin & axes definition + measured court dimensions.
- Checkerboard square size.
- Camera settings used.
- Each camera's measured position + height.
- Sync method.
- Marker coordinate list.

---

## 11. Pitfalls that silently ruin the dataset
- Changing zoom / focus / exposure mid-session → calibration void.
- A camera that can't see all court keypoints → uncalibratable → useless.
- Bumping a camera after its empty-court reference → re-shoot that reference.
- No checkerboard calibration → no undistortion / metric ever.
- No ground-truth markers → can't measure accuracy.
- No occlusion clips → not a redundancy dataset.
- Unpaired clip numbers → can't match the two views.
- HDR or variable frame rate → processing pain.
- Undocumented clips → unusable to the next team.

---

## 12. Pre-shoot checklist (run through before recording each session)
- [ ] All cameras: 60fps, 1080p, zoom 1.0×, AE/AF locked, HDR off, landscape.
- [ ] Intrinsic checkerboard captured for each camera (this session's settings).
- [ ] Cameras placed; each sees full near half + all 12 keypoints (verify on screen).
- [ ] Empty-court reference recorded per camera.
- [ ] Camera positions + heights measured and logged; setup photographed.
- [ ] Court dimensions measured; origin/axes written in README.
- [ ] Ground-truth markers placed and coordinates logged.
- [ ] Sync event procedure agreed; done at start of every clip.
- [ ] Naming scheme + master log open and ready.
```
