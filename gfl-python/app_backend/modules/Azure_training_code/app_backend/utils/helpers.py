import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from app_backend.config import LOWER_COLOR, UPPER_COLOR
import time
import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import shutil
from app_backend.config import IMG_SIZE, DB_FOLDER
from sklearn.decomposition import PCA


# def find_pixels_per_inch(img):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
#     if not contours:
#         raise ValueError("No neon marker found")
 
#     largest = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest)
#     return x, y, w, h
 
 
# def find_head_tail(crop):
#     original_h, original_w = crop.shape[:2]
 
#     # Resize only if crop is large
#     max_dim = max(original_h, original_w)
#     resize_factor = 600 / max_dim if max_dim > 600 else 1.0
 
#     if resize_factor < 1.0:
#         small_crop = cv2.resize(crop, (0, 0), fx=resize_factor, fy=resize_factor)
#     else:
#         small_crop = crop
 
#     mask = np.zeros(small_crop.shape[:2], np.uint8)
#     bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
 
#     rect = (5, 5, small_crop.shape[1]-10, small_crop.shape[0]-10)
#     cv2.grabCut(small_crop, mask, rect, bgd, fgd, 1, cv2.GC_INIT_WITH_RECT)
 
#     bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
 
#     if resize_factor < 1.0:
#         bin_mask = cv2.resize(bin_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
 
#     cleaned = crop * bin_mask[:, :, np.newaxis]
 
#     gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
 
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
#     if not contours:
#         return (0, 0), (0, 0), 0
 
#     cnt = max(contours, key=cv2.contourArea)
#     head, tail, max_dist = find_head_tail_optimized(cnt)
 
#     return head, tail, max_dist
 
 
# make sure you have this import somewhere:
# from sklearn.decomposition import PCA
 
# def find_head_tail_optimized(cnt):
#     """
#     PCA-based method with end disambiguation:
#     - Fit PCA to contour points for body axis.
#     - Compute thin slices at both extremes along the axis.
#     - The slice with larger perpendicular spread is the TAIL (caudal fin).
#     - HEAD (mouth) is the opposite extreme.
#     - Tail point = midpoint of dorsal/ventral extremes in the tail slice.
#     - Head point = most anterior extreme point in the head slice.
#     """
#     points = cnt.reshape(-1, 2).astype(np.float32)
#     if len(points) < 2:
#         return (0, 0), (0, 0), 0
 
#     # PCA axis
#     pca = PCA(n_components=2)
#     pca.fit(points)
#     axis = pca.components_[0]
#     axis = axis / (np.linalg.norm(axis) + 1e-8)
#     perp_axis = np.array([-axis[1], axis[0]], dtype=np.float32)
 
#     # Projections
#     proj = points @ axis
#     pmin, pmax = float(proj.min()), float(proj.max())
#     length_along = pmax - pmin
#     if length_along < 1e-3:
#         # Degenerate
#         p = tuple(points[0].astype(int))
#         return p, p, 0
 
#     # Use a scale-aware slice thickness (≈3% of length, min 3 px)
#     slice_th = max(3.0, 0.03 * length_along)
 
#     # Two end slices
#     A_idx = proj < (pmin + slice_th)   # "A" end near pmin
#     B_idx = proj > (pmax - slice_th)   # "B" end near pmax
#     A_pts = points[A_idx]
#     B_pts = points[B_idx]
 
#     # Fallbacks if a slice is empty
#     if len(A_pts) < 2 or len(B_pts) < 2:
#         # fall back to raw extremes
#         head_raw = points[np.argmin(proj)]
#         tail_raw = points[np.argmax(proj)]
#         head = tuple(np.round(head_raw).astype(int))
#         tail = tuple(np.round(tail_raw).astype(int))
#         max_dist = float(np.linalg.norm(head_raw - tail_raw))
#         return head, tail, max_dist
 
#     # Measure perpendicular spread (caudal fin is wider)
#     def slice_stats(pts):
#         pp = pts @ perp_axis
#         width = float(pp.max() - pp.min())
#         return width, pp
 
#     A_width, A_pp = slice_stats(A_pts)
#     B_width, B_pp = slice_stats(B_pts)
 
#     # Decide tail end by larger perpendicular spread
#     tail_is_A = A_width > B_width
 
#     if tail_is_A:
#         # Tail from A slice: midpoint of dorsal/ventral extremes
#         dorsal = A_pts[np.argmax(A_pp)]
#         ventral = A_pts[np.argmin(A_pp)]
#         tail_pt = (dorsal + ventral) / 2.0
#         # Head from opposite end: most anterior extreme there
#         head_pt = points[np.argmax(proj)]  # opposite extreme (near B)
#     else:
#         dorsal = B_pts[np.argmax(B_pp)]
#         ventral = B_pts[np.argmin(B_pp)]
#         tail_pt = (dorsal + ventral) / 2.0
#         head_pt = points[np.argmin(proj)]  # opposite extreme (near A)
 
#     head = tuple(np.round(head_pt).astype(int))
#     tail = tuple(np.round(tail_pt).astype(int))
#     max_dist = float(np.linalg.norm(np.array(head_pt) - np.array(tail_pt)))
 
#     return head, tail, max_dist
 
 
# def find_head_tail_optimized(cnt):
#     """
#     PCA-based method with end disambiguation:
#     - Fit PCA to contour points for body axis.
#     - Compute thin slices at both extremes along the axis.
#     - The slice with larger perpendicular spread is the TAIL (caudal fin).
#     - HEAD (mouth) is the opposite extreme.
#     - Tail point = center (midpoint) at the caudal FIN BASE (body-touching),
#       found by the strongest width jump near the tail side.
#     - Head point = most anterior extreme point on the head side.
#     """
#     points = cnt.reshape(-1, 2).astype(np.float32)
#     if len(points) < 2:
#         return (0, 0), (0, 0), 0
 
#     # PCA axis
#     pca = PCA(n_components=2)
#     pca.fit(points)
#     axis = pca.components_[0]
#     axis = axis / (np.linalg.norm(axis) + 1e-8)
#     perp_axis = np.array([-axis[1], axis[0]], dtype=np.float32)
 
#     # Projections
#     proj = points @ axis
#     pmin, pmax = float(proj.min()), float(proj.max())
#     length_along = pmax - pmin
#     if length_along < 1e-3:
#         p = tuple(points[0].astype(int))
#         return p, p, 0
 
#     # Use a scale-aware slice thickness (≈3% of length, min 3 px)
#     slice_th = max(3.0, 0.03 * length_along)
 
#     # Two end slices
#     A_idx = proj < (pmin + slice_th)   # "A" end near pmin
#     B_idx = proj > (pmax - slice_th)   # "B" end near pmax
#     A_pts = points[A_idx]
#     B_pts = points[B_idx]
 
#     # If a slice is empty, fall back to raw extremes
#     if len(A_pts) < 2 or len(B_pts) < 2:
#         head_raw = points[np.argmin(proj)]
#         tail_raw = points[np.argmax(proj)]
#         head = tuple(np.round(head_raw).astype(int))
#         tail = tuple(np.round(tail_raw).astype(int))
#         max_dist = float(np.linalg.norm(head_raw - tail_raw))
#         return head, tail, max_dist
 
#     # Measure perpendicular spread (caudal fin side is wider)
#     def slice_stats(pts):
#         pp = pts @ perp_axis
#         width = float(pp.max() - pp.min())
#         return width, pp
 
#     A_width, A_pp = slice_stats(A_pts)
#     B_width, B_pp = slice_stats(B_pts)
 
#     # Decide tail end by larger perpendicular spread (flip-invariant)
#     tail_is_A = A_width > B_width
 
#     # --- Build width profile along the axis (robust bins) ---
#     nbins = 80
#     bin_edges = np.linspace(pmin, pmax, nbins + 1)
#     widths = np.zeros(nbins, dtype=np.float32)
#     dorsal_idx = np.full(nbins, -1, dtype=int)
#     ventral_idx = np.full(nbins, -1, dtype=int)
 
#     pp_all = points @ perp_axis
#     for i in range(nbins):
#         # expanded slice to avoid empties (±1 bin)
#         lo = bin_edges[max(0, i - 1)]
#         hi = bin_edges[min(nbins, i + 2)]
#         sel = (proj >= lo) & (proj < hi)
#         if np.any(sel):
#             pps = pp_all[sel]
#             widths[i] = pps.max() - pps.min()
#             idxs = np.flatnonzero(sel)
#             dorsal_idx[i] = idxs[np.argmax(pps)]
#             ventral_idx[i] = idxs[np.argmin(pps)]
#         else:
#             widths[i] = 0.0
 
#     # --- From the tail side, find strongest positive jump in width = FIN BASE ---
#     span = max(6, nbins // 4)  # search only near the tail quarter
#     if tail_is_A:
#         seg = widths[:span]
#         diffs = np.diff(seg, prepend=seg[0])
#         base_bin = int(np.argmax(diffs))
#         head_pt_raw = points[np.argmax(proj)]  # head is opposite extreme (mouth tip)
#     else:
#         seg = widths[-span:]
#         diffs = np.diff(seg, prepend=seg[0])
#         base_bin = int(np.argmax(diffs)) + (nbins - span)
#         head_pt_raw = points[np.argmin(proj)]
 
#     # Fallback if indices missing: use simple extremes
#     if dorsal_idx[base_bin] == -1 or ventral_idx[base_bin] == -1:
#         head_raw = head_pt_raw
#         tail_raw = points[np.argmin(proj)] if not tail_is_A else points[np.argmax(proj)]
#         head = tuple(np.round(head_raw).astype(int))
#         tail = tuple(np.round(tail_raw).astype(int))
#         max_dist = float(np.linalg.norm(head_raw - tail_raw))
#         return head, tail, max_dist
 
#     # Tail point at FIN BASE = midpoint of dorsal/ventral (center, body-touching)
#     dorsal = points[dorsal_idx[base_bin]]
#     ventral = points[ventral_idx[base_bin]]
#     tail_pt_raw = (dorsal + ventral) / 2.0
 
#     head_pt = head_pt_raw
#     tail_pt = tail_pt_raw
 
#     head = tuple(np.round(head_pt).astype(int))
#     tail = tuple(np.round(tail_pt).astype(int))
#     max_dist = float(np.linalg.norm(np.array(head_pt) - np.array(tail_pt)))
 
#     return head, tail, max_dist
 
 
 
def find_pixels_per_inch(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    if not contours:
        raise ValueError("No neon marker found")
 
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h
 
 
def find_head_tail(crop):
    original_h, original_w = crop.shape[:2]
 
    # Resize only if crop is large
    max_dim = max(original_h, original_w)
    resize_factor = 600 / max_dim if max_dim > 600 else 1.0
 
    if resize_factor < 1.0:
        small_crop = cv2.resize(crop, (0, 0), fx=resize_factor, fy=resize_factor)
    else:
        small_crop = crop
 
    mask = np.zeros(small_crop.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
 
    rect = (5, 5, small_crop.shape[1]-10, small_crop.shape[0]-10)
    cv2.grabCut(small_crop, mask, rect, bgd, fgd, 1, cv2.GC_INIT_WITH_RECT)
 
    bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
 
    if resize_factor < 1.0:
        bin_mask = cv2.resize(bin_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
 
    cleaned = crop * bin_mask[:, :, np.newaxis]
 
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
 
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    if not contours:
        return (0, 0), (0, 0), 0
 
    cnt = max(contours, key=cv2.contourArea)
    head, tail, max_dist = find_head_tail_optimized(cnt)
 
    return head, tail, max_dist
 

 
def find_head_tail_optimized(cnt):
    """
    Mouth (HEAD) → Tail BASE (caudal peduncle) finder using only the contour.
 
    - PCA via NumPy SVD to get body axis.
    - Decide tail side by comparing perpendicular spread at the two ends.
    - HEAD point = raw extreme along the axis on the head side (more mouth-consistent).
    - Build a tight raster mask from the contour.
    - From the tail extreme, scan inward along the axis; pick the first slice
      where cross-section runs drop from >=2 to 1 ⇒ tail BASE.
      Fallback: choose the narrowest cross-section (peduncle).
    Returns:
        head_xy (int,int), tail_xy (int,int), max_dist (float)
    """
    # ---------- sanity & shape ----------
    if cnt is None:
        return (0, 0), (0, 0), 0.0
    pts = np.asarray(cnt, dtype=np.float64)
    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        P = pts[:, 0, :]
    elif pts.ndim == 2 and pts.shape[1] == 2:
        P = pts
    else:
        return (0, 0), (0, 0), 0.0
    if P.shape[0] < 5:
        p = tuple(np.round(P[0]).astype(int)) if P.size else (0, 0)
        return p, p, 0.0
 
    # ---------- PCA axis ----------
    C = P.mean(axis=0)
    X = P - C
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        axis = Vt[0]
    except Exception:
        axis = np.array([1.0, 0.0], dtype=np.float64)
    nrm = float(np.linalg.norm(axis))
    if not np.isfinite(nrm) or nrm < 1e-12:
        p = tuple(np.round(P[0]).astype(int))
        return p, p, 0.0
    axis = axis / nrm
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
 
    # ---------- projections ----------
    proj = (P - C) @ axis
    cross = (P - C) @ perp
    pmin, pmax = float(proj.min()), float(proj.max())
    L = pmax - pmin
    if L < 1e-6 or not np.isfinite(L):
        p = tuple(np.round(P[0]).astype(int))
        return p, p, 0.0
 
    # ---------- tail vs head side by end spread ----------
    end_th = max(2.0, 0.03 * L)  # ~3% of body length
    A_mask = proj <= (pmin + end_th)  # near min end
    B_mask = proj >= (pmax - end_th)  # near max end
    A_width = (cross[A_mask].max() - cross[A_mask].min()) if np.any(A_mask) else 0.0
    B_width = (cross[B_mask].max() - cross[B_mask].min()) if np.any(B_mask) else 0.0
    tail_is_A = A_width > B_width
 
    # ---------- HEAD = raw extreme at head side (mouth start) ----------
    if tail_is_A:
        head_pt = P[np.argmax(proj)]   # opposite of tail side
        tail_t  = pmin                 # tail extreme parameter
        head_t  = pmax
    else:
        head_pt = P[np.argmin(proj)]
        tail_t  = pmax
        head_t  = pmin
 
    # ---------- raster mask from contour (tight bbox + pad) ----------
    x, y, w, h = cv2.boundingRect(cnt)
    pad = 4
    off = np.array([x - pad, y - pad], dtype=np.int32)
    Hm, Wm = h + 2 * pad, w + 2 * pad
    mask = np.zeros((Hm, Wm), np.uint8)
    cnt_shift = (P - off).astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [cnt_shift], -1, 255, thickness=-1)
 
    # helper: runs & width of a perpendicular slice through world point q
    def runs_and_width(q_world, half_len):
        q = q_world - off.astype(np.float64)
        ts = np.linspace(-half_len, half_len, int(2 * half_len) + 1)
        xs = np.clip(q[0] + ts * perp[0], 0, Wm - 1).astype(int)
        ys = np.clip(q[1] + ts * perp[1], 0, Hm - 1).astype(int)
        line = (mask[ys, xs] > 0).astype(np.uint8)
        runs, width, in_run = 0, 0, False
        for v in line:
            width += int(v)
            if v and not in_run:
                runs += 1; in_run = True
            elif not v and in_run:
                in_run = False
        return runs, width
 
    # ---------- scan from tail extreme inward to find BASE ----------
    scan_back = max(60.0, 0.25 * L)           # how far to search inward
    steps     = int(max(120, 0.9 * scan_back))
    half_len  = max(40.0, 0.18 * L)           # slice half-length (≈ body depth)
 
    ts = np.linspace(tail_t, tail_t + (scan_back if tail_is_A else -scan_back), steps)
    prev_runs = None
    widths, pts_world = [], []
    tail_base = None
 
    for t in ts:
        q = C + t * axis
        r, wth = runs_and_width(q, half_len)
        pts_world.append(q); widths.append(wth)
        if prev_runs is not None and (prev_runs >= 2 and r == 1):
            tail_base = q
            break
        prev_runs = r
 
    if tail_base is None:
        # fallback: narrowest cross-section (peduncle)
        k = int(np.argmin(np.asarray(widths))) if widths else 0
        tail_base = pts_world[k] if pts_world else (C + tail_t * axis)
 
    # ---------- pack outputs ----------
    head = tuple(np.round(head_pt).astype(int))
    tail = tuple(np.round(tail_base).astype(int))
    max_dist = float(np.linalg.norm(
        np.asarray(head, dtype=np.float64) - np.asarray(tail, dtype=np.float64)
    ))
    return head, tail, max_dist
 
#  These are for Predict V2
def find_head_tail_optimized_v2(cnt):
    """
    Mouth (HEAD) → Tail BASE (caudal peduncle) finder using only the contour.
 
    - PCA via NumPy SVD to get body axis.
    - Decide tail side by comparing perpendicular spread at the two ends.
    - HEAD point = raw extreme along the axis on the head side (more mouth-consistent).
    - Build a tight raster mask from the contour.
    - From the tail extreme, scan inward along the axis; pick the first slice
      where cross-section runs drop from >=2 to 1 ⇒ tail BASE.
      Fallback: choose the narrowest cross-section (peduncle).
    Returns:
        head_xy (int,int), tail_xy (int,int), max_dist (float)
    """
    # ---------- sanity & shape ----------
    if cnt is None:
        return (0, 0), (0, 0), 0.0
    pts = np.asarray(cnt, dtype=np.float64)
    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        P = pts[:, 0, :]
    elif pts.ndim == 2 and pts.shape[1] == 2:
        P = pts
    else:
        return (0, 0), (0, 0), 0.0
    if P.shape[0] < 5:
        p = tuple(np.round(P[0]).astype(int)) if P.size else (0, 0)
        return p, p, 0.0
 
    # ---------- PCA axis ----------
    C = P.mean(axis=0)
    X = P - C
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        axis = Vt[0]
    except Exception:
        axis = np.array([1.0, 0.0], dtype=np.float64)
    nrm = float(np.linalg.norm(axis))
    if not np.isfinite(nrm) or nrm < 1e-12:
        p = tuple(np.round(P[0]).astype(int))
        return p, p, 0.0
    axis = axis / nrm
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
 
    # ---------- projections ----------
    proj = (P - C) @ axis
    cross = (P - C) @ perp
    pmin, pmax = float(proj.min()), float(proj.max())
    L = pmax - pmin
    if L < 1e-6 or not np.isfinite(L):
        p = tuple(np.round(P[0]).astype(int))
        return p, p, 0.0
 
    # ---------- tail vs head side by end spread ----------
    end_th = max(2.0, 0.03 * L)  # ~3% of body length
    A_mask = proj <= (pmin + end_th)  # near min end
    B_mask = proj >= (pmax - end_th)  # near max end
    A_width = (cross[A_mask].max() - cross[A_mask].min()) if np.any(A_mask) else 0.0
    B_width = (cross[B_mask].max() - cross[B_mask].min()) if np.any(B_mask) else 0.0
    tail_is_A = A_width > B_width
 
    # ---------- HEAD = raw extreme at head side (mouth start) ----------
    if tail_is_A:
        head_pt = P[np.argmax(proj)]   # opposite of tail side
        tail_t  = pmin                 # tail extreme parameter
        head_t  = pmax
    else:
        head_pt = P[np.argmin(proj)]
        tail_t  = pmax
        head_t  = pmin
 
    # ---------- raster mask from contour (tight bbox + pad) ----------
    x, y, w, h = cv2.boundingRect(cnt)
    pad = 4
    off = np.array([x - pad, y - pad], dtype=np.int32)
    Hm, Wm = h + 2 * pad, w + 2 * pad
    mask = np.zeros((Hm, Wm), np.uint8)
    cnt_shift = (P - off).astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [cnt_shift], -1, 255, thickness=-1)
 
    # helper: runs & width of a perpendicular slice through world point q
    def runs_and_width(q_world, half_len):
        q = q_world - off.astype(np.float64)
        ts = np.linspace(-half_len, half_len, int(2 * half_len) + 1)
        xs = np.clip(q[0] + ts * perp[0], 0, Wm - 1).astype(int)
        ys = np.clip(q[1] + ts * perp[1], 0, Hm - 1).astype(int)
        line = (mask[ys, xs] > 0).astype(np.uint8)
        runs, width, in_run = 0, 0, False
        for v in line:
            width += int(v)
            if v and not in_run:
                runs += 1; in_run = True
            elif not v and in_run:
                in_run = False
        return runs, width
 
    # ---------- scan from tail extreme inward to find BASE ----------
    scan_back = max(60.0, 0.25 * L)           # how far to search inward
    steps     = int(max(120, 0.9 * scan_back))
    half_len  = max(40.0, 0.18 * L)           # slice half-length (≈ body depth)
 
    ts = np.linspace(tail_t, tail_t + (scan_back if tail_is_A else -scan_back), steps)
    prev_runs = None
    widths, pts_world = [], []
    tail_base = None
 
    for t in ts:
        q = C + t * axis
        r, wth = runs_and_width(q, half_len)
        pts_world.append(q); widths.append(wth)
        if prev_runs is not None and (prev_runs >= 2 and r == 1):
            tail_base = q
            break
        prev_runs = r
 
    if tail_base is None:
        # fallback: narrowest cross-section (peduncle)
        k = int(np.argmin(np.asarray(widths))) if widths else 0
        tail_base = pts_world[k] if pts_world else (C + tail_t * axis)
 
    # ---------- pack outputs ----------
    head = tuple(np.round(head_pt).astype(int))
    tail = tuple(np.round(tail_base).astype(int))
    max_dist = float(np.linalg.norm(
        np.asarray(head, dtype=np.float64) - np.asarray(tail, dtype=np.float64)
    ))
    return head, tail, max_dist
 
 
def find_pixels_per_inch_v2(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    if not contours:
        raise ValueError("No neon marker found")
 
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h
 
 
def find_head_tail_v2(crop):
    original_h, original_w = crop.shape[:2]
 
    # Resize only if crop is large
    max_dim = max(original_h, original_w)
    resize_factor = 600 / max_dim if max_dim > 600 else 1.0
 
    if resize_factor < 1.0:
        small_crop = cv2.resize(crop, (0, 0), fx=resize_factor, fy=resize_factor)
    else:
        small_crop = crop
 
    mask = np.zeros(small_crop.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
 
    rect = (5, 5, small_crop.shape[1]-10, small_crop.shape[0]-10)
    cv2.grabCut(small_crop, mask, rect, bgd, fgd, 1, cv2.GC_INIT_WITH_RECT)
 
    bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
 
    if resize_factor < 1.0:
        bin_mask = cv2.resize(bin_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
 
    cleaned = crop * bin_mask[:, :, np.newaxis]
 
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
 
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    if not contours:
        return (0, 0), (0, 0), 0
 
    cnt = max(contours, key=cv2.contourArea)
    head, tail, max_dist = find_head_tail_optimized_v2(cnt)
 
    return head, tail, max_dist


  
 
 


# make sure you have this import somewhere:
# from sklearn.decomposition import PCA

def find_pixels_per_inch(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No neon marker found")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


# def find_head_tail(crop):
#     # start_time = time.time()
#     mask = np.zeros(crop.shape[:2], np.uint8)
#     # print("mask: ", time.time() - start_time)

#     bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
#     # print("bgd: ", time.time() - start_time)

#     rect = (5, 5, crop.shape[1]-10, crop.shape[0]-10)
#     # print("rect: ", time.time() - start_time)

#     cv2.grabCut(crop, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

#     # print("cv2.grabCut: ", time.time() - start_time)

#     bin_mask = np.where((mask==2)|(mask==0), 0, 1).astype("uint8")
#     # print("bin_mask: ", time.time() - start_time)

#     cleaned = crop * bin_mask[:, :, np.newaxis]
#     # print("cleaned: ", time.time() - start_time)

#     gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
#     # print("gray: ", time.time() - start_time)

#     _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#     # print("binary: ", time.time() - start_time)

#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # print("contours ", time.time() - start_time)

#     if not contours:
#         return (0, 0), (0, 0), 0
#     cnt = max(contours, key=cv2.contourArea)
#     head, tail, max_dist = find_head_tail_optimized(cnt)

#     # max_dist, head, tail = 0, (0, 0), (0, 0)
#     # print("cnt: ", time.time() - start_time)

#     # for i in range(len(cnt)):
#     #     for j in range(i+1, len(cnt)):
#     #         pt1 = tuple(cnt[i][0])
#     #         pt2 = tuple(cnt[j][0])
#     #         dist = euclidean(pt1, pt2)
#     #         if dist > max_dist:
#     #             max_dist = dist
#     #             head, tail = pt1, pt2


#     # print("return : ", time.time() - start_time)

#     return head, tail, max_dist

# # #optimized by reducing the crop image resolution and reduce Grabcut iteration


def find_head_tail(crop):
    original_h, original_w = crop.shape[:2]

    # Resize only if crop is large
    max_dim = max(original_h, original_w)
    resize_factor = 600 / max_dim if max_dim > 600 else 1.0

    if resize_factor < 1.0:
        small_crop = cv2.resize(crop, (0, 0), fx=resize_factor, fy=resize_factor)
    else:
        small_crop = crop

    mask = np.zeros(small_crop.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)

    rect = (5, 5, small_crop.shape[1]-10, small_crop.shape[0]-10)
    cv2.grabCut(small_crop, mask, rect, bgd, fgd, 1, cv2.GC_INIT_WITH_RECT)

    bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    if resize_factor < 1.0:
        bin_mask = cv2.resize(bin_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    cleaned = crop * bin_mask[:, :, np.newaxis]

    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return (0, 0), (0, 0), 0

    cnt = max(contours, key=cv2.contourArea)
    head, tail, max_dist = find_head_tail_optimized(cnt)

    return head, tail, max_dist


def find_head_tail_optimized(cnt):
    hull = cv2.convexHull(cnt, returnPoints=True)
    n = len(hull)

    if n < 2:
        return (0, 0), (0, 0), 0

    max_dist = 0
    head = tail = (0, 0)

    j = 1
    for i in range(n):
        while True:
            curr = euclidean(hull[i][0], hull[j % n][0])
            next_j = euclidean(hull[i][0], hull[(j + 1) % n][0])
            if next_j > curr:
                j += 1
            else:
                break
        dist = euclidean(hull[i][0], hull[j % n][0])
        if dist > max_dist:
            max_dist = dist
            head = tuple(hull[i][0])
            tail = tuple(hull[j % n][0])

    return head, tail, max_dist



# uniqueness

def ensure_db_folder():
    os.makedirs(DB_FOLDER, exist_ok=True)

def apply_clahe_and_sharpening(pil_image):
    img_array = np.array(pil_image.convert("RGB")).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_bgr = cv2.filter2D(enhanced_bgr, -1, kernel)
    sharpened_rgb = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(sharpened_rgb)

def preprocess_image(img_path):
    try:
        raw_img = Image.open(img_path).convert("RGB")
        enhanced_img = apply_clahe_and_sharpening(raw_img)
        enhanced_img = enhanced_img.resize(IMG_SIZE)
        img_array = img_to_array(enhanced_img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None




