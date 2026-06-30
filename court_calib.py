"""
Pickleball near-court ROI via homography + RANSAC.

Pipeline:
  detected 12 image keypoints  ->  RANSAC homography to canonical court (feet)
  ->  bisect in court space (exact net line)  ->  inverse-project near half
  ->  back to image  ->  ROI polygon + (optional) line-refine + re-projection error

Develop/test on ground-truth label points first; swap in model predictions later.

Keypoint numbering (from the project diagram):
   0 far baseline L   1 far baseline mid   2 far baseline R
   3 far kitchen R    4 far kitchen mid    5 far kitchen L
   6 near kitchen L   7 near kitchen mid   8 near kitchen R
   9 near baseline R  10 near baseline mid 11 near baseline L
Visibility: 2 visible / 1 occluded (position known) / 0 not labeled.
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 1. CANONICAL COURT MODEL  (real pickleball dimensions, in feet)
#    Origin (0,0) at far-baseline-left. x across court (0..20), y along court (0..44).
#    Net at y = 22. Kitchen lines at y = 15 (far) and y = 29 (near).
# ---------------------------------------------------------------------------
COURT_W = 20.0
COURT_L = 44.0
NET_Y   = 22.0
KITCHEN = 7.0

COURT_PTS = {
    0:  (0.0,         0.0),
    1:  (COURT_W / 2, 0.0),
    2:  (COURT_W,     0.0),
    3:  (COURT_W,     NET_Y - KITCHEN),   # y = 15
    4:  (COURT_W / 2, NET_Y - KITCHEN),
    5:  (0.0,         NET_Y - KITCHEN),
    6:  (0.0,         NET_Y + KITCHEN),   # y = 29
    7:  (COURT_W / 2, NET_Y + KITCHEN),
    8:  (COURT_W,     NET_Y + KITCHEN),
    9:  (COURT_W,     COURT_L),           # y = 44
    10: (COURT_W / 2, COURT_L),
    11: (0.0,         COURT_L),
}


# ---------------------------------------------------------------------------
# 2. HOMOGRAPHY via RANSAC  (image -> court, and inverse)
# ---------------------------------------------------------------------------
def compute_homography(keypoints, ransac_thresh_px=5.0):
    """
    keypoints: list of 12 (x, y, v) in image pixels.
    Uses all points with v>=1 (visible + occluded). RANSAC rejects outliers.
    Returns (H_img2court, H_court2img, inlier_mask, used_indices).
    """
    img_pts, court_pts, idxs = [], [], []
    for i, kp in enumerate(keypoints):
        x, y = kp[0], kp[1]
        v = kp[2] if len(kp) > 2 else 2
        if v >= 1 and i in COURT_PTS:
            img_pts.append([x, y])
            court_pts.append(COURT_PTS[i])
            idxs.append(i)

    if len(img_pts) < 4:
        raise ValueError(f"need >=4 usable keypoints, got {len(img_pts)}")

    img_pts   = np.asarray(img_pts,   dtype=np.float64)
    court_pts = np.asarray(court_pts, dtype=np.float64)

    H, mask = cv2.findHomography(img_pts, court_pts,
                                 method=cv2.RANSAC,
                                 ransacReprojThreshold=ransac_thresh_px)
    if H is None:
        raise ValueError("homography failed (degenerate / too few inliers)")
    return H, np.linalg.inv(H), mask.ravel().astype(bool), idxs


# ---------------------------------------------------------------------------
# 3. RE-PROJECTION ERROR  (validation metric, in feet and pixels)
# ---------------------------------------------------------------------------
def reprojection_error(keypoints, H, H_inv, used_indices, inlier_mask=None):
    errs_ft, errs_px = [], []
    for j, i in enumerate(used_indices):
        if inlier_mask is not None and not inlier_mask[j]:
            continue  # measure error only on points RANSAC trusted
        kp = keypoints[i]
        img_pt = np.array([kp[0], kp[1], 1.0])

        c = H @ img_pt; c /= c[2]
        gt_c = np.array(COURT_PTS[i])
        errs_ft.append(np.linalg.norm(c[:2] - gt_c))

        gt_h = np.array([gt_c[0], gt_c[1], 1.0])
        p = H_inv @ gt_h; p /= p[2]
        errs_px.append(np.linalg.norm(p[:2] - img_pt[:2]))

    if not errs_px:
        return {"mean_px": float("nan"), "max_px": float("nan"),
                "mean_ft": float("nan"), "n_points": 0}
    return {
        "mean_px": float(np.mean(errs_px)),
        "max_px":  float(np.max(errs_px)),
        "mean_ft": float(np.mean(errs_ft)),
        "n_points": len(errs_px),
    }


# ---------------------------------------------------------------------------
# 4. NEAR-HALF ROI  (court y in [NET_Y, COURT_L]) projected back to image
# ---------------------------------------------------------------------------
def _to_img(H_inv, cx, cy):
    v = H_inv @ np.array([cx, cy, 1.0]); v /= v[2]
    return [v[0], v[1]]


def near_half_polygon(H_inv, samples_per_edge=20):
    """Image-space polygon (Nx2 int) of the near half, sampled along edges so the
    net line + sidelines come back correctly curved under perspective."""
    pts = []
    for t in np.linspace(0, 1, samples_per_edge):                 # net: L->R
        pts.append(_to_img(H_inv, t * COURT_W, NET_Y))
    for t in np.linspace(0, 1, samples_per_edge):                 # right sideline
        pts.append(_to_img(H_inv, COURT_W, NET_Y + t * (COURT_L - NET_Y)))
    for t in np.linspace(0, 1, samples_per_edge):                 # near baseline R->L
        pts.append(_to_img(H_inv, COURT_W - t * COURT_W, COURT_L))
    for t in np.linspace(0, 1, samples_per_edge):                 # left sideline
        pts.append(_to_img(H_inv, 0.0, COURT_L - t * (COURT_L - NET_Y)))
    return np.array(pts, dtype=np.int32)


def net_line_image(H_inv):
    l = _to_img(H_inv, 0.0, NET_Y)
    r = _to_img(H_inv, COURT_W, NET_Y)
    return (int(l[0]), int(l[1])), (int(r[0]), int(r[1]))


def near_half_wireframe(H_inv):
    """
    Near-half wireframe in IMAGE coords, computed from the homography so derived
    net points are exact. Returns {'dots': [[x,y],...], 'lines': [[[x,y],[x,y]],...]}.

    Dots: net L / net M / net R (derived), kitchen 6/7/8, baseline 9/10/11.
    Lines: net (L-R), left side (netL->base11), right side (netR->base9),
           kitchen (6-8), center serve line kitchen->baseline (7-10).
    """
    def P(cx, cy):
        v = _to_img(H_inv, cx, cy)
        return [int(round(v[0])), int(round(v[1]))]

    # court-space coords
    netL, netM, netR = P(0, NET_Y), P(COURT_W / 2, NET_Y), P(COURT_W, NET_Y)
    k6 = P(0, NET_Y + KITCHEN)            # fg kitchen L
    k7 = P(COURT_W / 2, NET_Y + KITCHEN)  # fg kitchen M
    k8 = P(COURT_W, NET_Y + KITCHEN)      # fg kitchen R
    b11 = P(0, COURT_L)                   # near baseline L
    b10 = P(COURT_W / 2, COURT_L)         # near baseline M
    b9 = P(COURT_W, COURT_L)              # near baseline R

    dots = [netL, netM, netR, k6, k7, k8, b9, b10, b11]
    lines = [
        [netL, netR],   # net line
        [netL, b11],    # left side of near ROI
        [netR, b9],     # right side of near ROI
        [b11, b9],      # near baseline (left -> right)
        [k6, k8],       # kitchen line
        [k7, b10],      # center serve line: kitchen-mid -> baseline-mid
    ]
    return {"dots": dots, "lines": lines}


# ---------------------------------------------------------------------------
# 5. OPTIONAL net LINE-REFINE  (snap to brightest nearby pixels)
# ---------------------------------------------------------------------------
def refine_net_line(image_bgr, p_left, p_right, search_px=15):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    d = np.array(p_right, float) - np.array(p_left, float)
    L = np.linalg.norm(d)
    if L < 1:
        return p_left, p_right
    n = np.array([-(d[1] / L), d[0] / L])
    best_off, best_score = 0.0, -1.0
    for off in np.linspace(-search_px, search_px, 2 * search_px + 1):
        score = 0.0
        for t in np.linspace(0, 1, 50):
            base = np.array(p_left, float) + t * (np.array(p_right, float) - np.array(p_left, float))
            px = base + off * n
            xi, yi = int(round(px[0])), int(round(px[1]))
            if 0 <= yi < gray.shape[0] and 0 <= xi < gray.shape[1]:
                score += gray[yi, xi]
        if score > best_score:
            best_score, best_off = score, off
    shift = best_off * n
    return (tuple((np.array(p_left, float) + shift).astype(int)),
            tuple((np.array(p_right, float) + shift).astype(int)))


# ---------------------------------------------------------------------------
# 6. DRAW / MASK
# ---------------------------------------------------------------------------
def draw_near_half(image_bgr, keypoints, ransac_thresh_px=5.0,
                   do_refine=False, alpha=0.35):
    H, H_inv, inliers, used = compute_homography(keypoints, ransac_thresh_px)
    metrics = reprojection_error(keypoints, H, H_inv, used, inliers)

    poly = near_half_polygon(H_inv)
    p_l, p_r = net_line_image(H_inv)
    if do_refine:
        p_l, p_r = refine_net_line(image_bgr, p_l, p_r)

    out = image_bgr.copy()
    overlay = out.copy()
    cv2.fillPoly(overlay, [poly], (0, 200, 0))
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    cv2.polylines(out, [poly], True, (0, 200, 0), 2)
    cv2.line(out, p_l, p_r, (0, 0, 255), 3)

    for j, i in enumerate(used):
        kp = keypoints[i]
        col = (0, 255, 0) if inliers[j] else (0, 0, 255)
        cv2.circle(out, (int(kp[0]), int(kp[1])), 5, col, -1)
        cv2.putText(out, str(i), (int(kp[0]) + 6, int(kp[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out, metrics, poly


def near_half_mask(image_shape, keypoints, ransac_thresh_px=5.0):
    _, H_inv, _, _ = compute_homography(keypoints, ransac_thresh_px)
    poly = near_half_polygon(H_inv)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


# ---------------------------------------------------------------------------
# 7. POINT LOADERS
# ---------------------------------------------------------------------------
def load_label_points(txt_path, img_w, img_h):
    """One YOLO-pose label line -> list of 12 (x, y, v) in pixels."""
    line = open(txt_path).read().strip().split("\n")[0].split()
    vals = list(map(float, line[5:]))
    pts = []
    for i in range(0, len(vals), 3):
        pts.append((vals[i] * img_w, vals[i + 1] * img_h, int(round(vals[i + 2]))))
    return pts


def model_points_from_result(result, vis_conf=0.5):
    """Ultralytics result[0] -> list of 12 (x, y, v). Use after training."""
    xy = result.keypoints.xy[0].cpu().numpy()
    conf = result.keypoints.conf[0].cpu().numpy()
    return [(float(xy[i, 0]), float(xy[i, 1]), 2 if conf[i] > vis_conf else 1)
            for i in range(len(xy))]
