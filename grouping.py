# grouping.py
import math
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree

import folium
from folium.plugins import Search


# =========================================================
# Parsing / Validation
# =========================================================
def _to_float_series(s: pd.Series) -> pd.Series:
    # supports decimal comma: "107,123" -> "107.123"
    s2 = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def validate_input_df(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return False, "File kosong.", None

    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    c_name = pick(["nama_toko", "nama toko", "toko", "store", "outlet", "nama_outlet", "nama outlet", "name"])
    c_lat = pick(["lat", "latitude", "y"])
    c_lon = pick(["long", "lon", "longitude", "lng", "x"])

    if not c_name or not c_lat or not c_lon:
        return (
            False,
            "Kolom wajib tidak ditemukan.\n\nWajib ada: **nama_toko**, **lat**, **long**.",
            None,
        )

    df2 = df[[c_name, c_lat, c_lon]].copy()
    df2.columns = ["nama_toko", "lat", "long"]

    df2["nama_toko"] = df2["nama_toko"].astype(str).str.strip()
    df2["lat"] = _to_float_series(df2["lat"])
    df2["long"] = _to_float_series(df2["long"])

    # attach stable row_id from original order
    df2["_row_id"] = np.arange(len(df2), dtype=int)

    bad = df2["nama_toko"].eq("") | df2["lat"].isna() | df2["long"].isna()
    bad_n = int(bad.sum())
    df2 = df2.loc[~bad].copy()

    if len(df2) == 0:
        return False, "Semua baris invalid setelah parsing lat/long. Pastikan lat/long angka.", None

    out_range = (df2["lat"].abs() > 90) | (df2["long"].abs() > 180)
    if out_range.any():
        ex = df2.loc[out_range].head(8)
        return (
            False,
            "Ada lat/long di luar range valid.\nLat harus -90..90, Long harus -180..180.\n\n"
            f"Contoh baris bermasalah:\n{ex.to_string(index=False)}",
            None,
        )

    return True, f"OK. Baris valid: {len(df2):,}. Baris dibuang: {bad_n:,}.", df2


# =========================================================
# Labels + Colors + Legend
# =========================================================
def label_from_gidx(gidx: int) -> str:
    return f"R{int(gidx) + 1:02d}"


def parse_label_to_gidx(label: str) -> int:
    # "R11" -> 10
    return int(label.replace("R", "")) - 1


def _palette(n: int):
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#0b3d91", "#ff1493",
        "#228b22", "#ff4500", "#6a5acd", "#2f4f4f", "#b8860b", "#008080",
        "#483d8b", "#a52a2a",
    ]
    return [base[i % len(base)] for i in range(n)]


def _add_legend(m: folium.Map, K: int, colors):
    items = "".join(
        f"""
        <div style="display:flex;align-items:center;margin-bottom:4px;">
          <div style="width:12px;height:12px;background:{colors[i]};margin-right:8px;border:1px solid #555;"></div>
          <div style="font-size:12px;">{label_from_gidx(i)}</div>
        </div>
        """
        for i in range(K)
    )
    legend_html = f"""
    <div style="
      position: fixed;
      bottom: 18px;
      left: 18px;
      z-index: 9999;
      background: rgba(255,255,255,0.92);
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #ddd;
      max-height: 260px;
      overflow: auto;
      box-shadow: 0 4px 14px rgba(0,0,0,0.12);
      ">
      <div style="font-weight:600;margin-bottom:8px;">Legend (Group)</div>
      {items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# =========================================================
# Convex Hull (Monotonic Chain) — no extra deps
# points: (lon, lat)
# =========================================================
def _convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# =========================================================
# Core math helpers
# =========================================================
def _compute_centroids(coords_deg: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    centroids = np.zeros((K, 2), dtype=float)
    for g in range(K):
        idx = np.where(labels == g)[0]
        if len(idx) == 0:
            centroids[g] = coords_deg.mean(axis=0)
        else:
            centroids[g] = coords_deg[idx].mean(axis=0)
    return centroids


def _initial_centroids_kmeans(coords_deg: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init="auto", batch_size=1024)
    labels0 = km.fit_predict(coords_deg.astype(float))
    return _compute_centroids(coords_deg, labels0, K)


def _balanced_assign(coords_deg: np.ndarray, centroids: np.ndarray, K: int, cap: int, soft_target: int):
    """
    Assign each point to nearest centroid with capacity.
    Try to keep <= soft_target first (balanced), then allow up to cap.
    """
    n = len(coords_deg)
    d = np.linalg.norm(coords_deg[:, None, :] - centroids[None, :, :], axis=2)

    # order points by confidence (gap between 1st and 2nd nearest)
    part = np.partition(d, 1, axis=1)
    gap = part[:, 1] - part[:, 0]
    order = np.argsort(gap)[::-1]

    labels = -np.ones(n, dtype=int)
    counts = np.zeros(K, dtype=int)

    for i in order:
        choices = np.argsort(d[i])
        placed = False

        # keep balanced first
        for g in choices:
            if counts[g] < soft_target and counts[g] < cap:
                labels[i] = g
                counts[g] += 1
                placed = True
                break

        if placed:
            continue

        # allow up to cap
        for g in choices:
            if counts[g] < cap:
                labels[i] = g
                counts[g] += 1
                placed = True
                break

        if not placed:
            raise RuntimeError("Tidak ada slot group tersisa. Pastikan K*cap >= n.")

    return labels


# =========================================================
# Apply overrides (LOCKED)
# =========================================================
def apply_overrides(df: pd.DataFrame, override_map: dict, K: int, cap: int):
    """
    override_map: {_row_id: gidx_target}
    - Enforce cap: if target is full, skip that override and report.
    - Mark locked rows with _locked = True
    """
    df = df.copy()
    df["_locked"] = False

    if not override_map:
        return df, [], []

    skipped = []
    applied = []

    # counts current
    counts = df["_gidx"].value_counts().to_dict()

    for rid, tgt in override_map.items():
        try:
            tgt = int(tgt)
        except Exception:
            skipped.append((rid, "invalid target"))
            continue

        if tgt < 0 or tgt >= K:
            skipped.append((rid, "target out of range"))
            continue

        rows = df.index[df["_row_id"] == int(rid)]
        if len(rows) == 0:
            skipped.append((rid, "row_id not found"))
            continue

        ix = rows[0]
        cur = int(df.loc[ix, "_gidx"])

        if cur == tgt:
            df.loc[ix, "_locked"] = True
            applied.append((rid, cur, tgt, "already"))
            continue

        tgt_count = counts.get(tgt, 0)
        if tgt_count >= cap:
            skipped.append((rid, f"target {label_from_gidx(tgt)} full"))
            continue

        # move
        df.loc[ix, "_gidx"] = tgt
        df.loc[ix, "_locked"] = True

        counts[cur] = counts.get(cur, 0) - 1
        counts[tgt] = counts.get(tgt, 0) + 1

        applied.append((rid, cur, tgt, "moved"))

    return df, applied, skipped


# =========================================================
# Refinement (optional) — DO NOT MOVE locked points
# =========================================================
def refine_local_knn(
    coords_deg: np.ndarray,
    labels: np.ndarray,
    locked_mask: np.ndarray,
    K: int,
    cap: int,
    refine_iter: int = 6,
    neighbor_k: int = 10,
    seed: int = 42,
):
    """
    Iterative local refinement:
    - kNN (haversine)
    - vote by neighbors (weighted inverse distance)
    - move only if improves distance-to-centroid
    - do NOT move locked points
    """
    if refine_iter <= 0:
        return labels

    rng = np.random.default_rng(seed)
    n = len(labels)

    coords_rad = np.radians(coords_deg)
    tree = BallTree(coords_rad, metric="haversine")

    kq = min(max(2, neighbor_k + 1), n)
    dist_rad, nbrs = tree.query(coords_rad, k=kq)
    dist_km = dist_rad * 6371.0088

    labels = labels.copy()
    counts = np.bincount(labels, minlength=K)

    for it in range(refine_iter):
        centroids = _compute_centroids(coords_deg, labels, K)
        order = np.arange(n)
        rng.shuffle(order)

        moved = 0

        for i in order:
            if locked_mask[i]:
                continue

            gi = labels[i]
            if counts[gi] <= 1:
                continue

            nb_idx = nbrs[i, 1:]
            nb_dist = dist_km[i, 1:]

            scores = np.zeros(K, dtype=float)
            w = 1.0 / (nb_dist + 1e-6)
            nb_lab = labels[nb_idx]
            for lab, ww in zip(nb_lab, w):
                scores[int(lab)] += ww

            cand = np.argsort(scores)[::-1]

            p = coords_deg[i]
            cur_d = np.linalg.norm(p - centroids[gi])

            best_g = gi
            best_d = cur_d

            for g in cand[:6]:
                if g == gi:
                    continue
                if counts[g] >= cap:
                    continue
                # optional: don't move into a cluster if it would immediately push out locked points (not needed here)
                new_d = np.linalg.norm(p - centroids[g])
                if new_d + 1e-9 < best_d:
                    best_g, best_d = g, new_d

            if best_g != gi:
                labels[i] = best_g
                counts[gi] -= 1
                counts[best_g] += 1
                moved += 1

        print(f"[REFINE] iter {it+1}/{refine_iter}: moved={moved}")
        if moved == 0:
            break

    return labels


# =========================================================
# Map builder (hull + colors + gmaps popup + legend)
# =========================================================
def build_map(df: pd.DataFrame, K: int):
    center = [df["lat"].mean(), df["long"].mean()]
    m = folium.Map(location=center, zoom_start=11, control_scale=True, tiles="OpenStreetMap")

    colors = _palette(K)

    # hull polygons first
    for gi in range(K):
        sub = df[df["_gidx"] == gi]
        if len(sub) < 3:
            continue
        pts = list(zip(sub["long"].tolist(), sub["lat"].tolist()))  # (lon, lat)
        hull = _convex_hull(pts)
        if len(hull) < 3:
            continue
        poly_latlon = [(y, x) for (x, y) in hull]
        folium.Polygon(
            locations=poly_latlon,
            color=colors[gi],
            weight=2,
            fill=True,
            fill_color=colors[gi],
            fill_opacity=0.15,
            opacity=0.9,
        ).add_to(m)

    marker_layer = folium.FeatureGroup(name="Toko", show=True).add_to(m)

    for _, r in df.iterrows():
        gi = int(r["_gidx"])
        code = label_from_gidx(gi)
        gmaps = f"https://www.google.com/maps?q={r['lat']},{r['long']}"
        lock_badge = " 🔒" if bool(r.get("_locked", False)) else ""

        popup_html = f"""
        <div style="font-size:13px;">
          <b>{r['nama_toko']}</b><br>
          Group: <b>{code}</b>{lock_badge}<br>
          Row ID: {int(r['_row_id'])}<br><br>
          <a href="{gmaps}" target="_blank">📍 Buka di Google Maps</a>
        </div>
        """

        folium.CircleMarker(
            location=[r["lat"], r["long"]],
            radius=5,
            color=colors[gi],
            fill=True,
            fill_color=colors[gi],
            fill_opacity=0.9,
            tooltip=f"[{int(r['_row_id']):04d}] {r['nama_toko']} ({code})",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(marker_layer)

    Search(
        layer=marker_layer,
        search_label="tooltip",
        placeholder="Cari nama toko / row id ...",
        collapsed=False,
    ).add_to(m)

    _add_legend(m, K, colors)
    folium.LayerControl(collapsed=True).add_to(m)

    # fit bounds
    sw = [df["lat"].min(), df["long"].min()]
    ne = [df["lat"].max(), df["long"].max()]
    m.fit_bounds([sw, ne], padding=(20, 20))
    return m


# =========================================================
# Pipeline API
# =========================================================
def run_grouping_pipeline(
    df_clean: pd.DataFrame,
    K: int,
    cap: int,
    override_map: dict | None = None,
    refine_on: bool = True,
    refine_iter: int = 6,
    neighbor_k: int = 10,
    seed: int = 42,
):
    """
    Returns:
      df_result: includes _row_id, _gidx, kategori, _locked
      folium_map
      applied_overrides, skipped_overrides
    """
    dfw = df_clean.copy().reset_index(drop=True)
    n = len(dfw)

    if K < 2:
        raise ValueError("K minimal 2.")
    if cap < 2:
        raise ValueError("cap minimal 2.")
    if K * cap < n:
        raise ValueError(f"K*cap tidak cukup. K={K}, cap={cap}, total={n}.")

    coords = dfw[["lat", "long"]].to_numpy(dtype=float)

    soft_target = int(math.ceil(n / K))
    soft_target = min(soft_target, int(cap))

    print(f"[INFO] n={n} | K={K} | cap={cap} | soft_target≈{soft_target} | refine_on={refine_on} | refine_iter={refine_iter} | neighbor_k={neighbor_k}")

    # 1) init centroids via kmeans, then balanced assignment
    centroids = _initial_centroids_kmeans(coords, K=K, seed=seed)
    labels = _balanced_assign(coords, centroids, K=K, cap=int(cap), soft_target=int(soft_target))

    dfw["_gidx"] = labels.astype(int)

    # 2) apply overrides (lock)
    override_map = override_map or {}
    dfw, applied, skipped = apply_overrides(dfw, override_map, K=K, cap=int(cap))

    # 3) refine (but do not move locked)
    if refine_on and refine_iter > 0:
        locked_mask = dfw["_locked"].to_numpy(dtype=bool)
        labels2 = refine_local_knn(
            coords_deg=coords,
            labels=dfw["_gidx"].to_numpy(dtype=int),
            locked_mask=locked_mask,
            K=K,
            cap=int(cap),
            refine_iter=int(refine_iter),
            neighbor_k=int(neighbor_k),
            seed=seed,
        )
        dfw["_gidx"] = labels2.astype(int)

        # 4) re-apply overrides to guarantee "override wins last"
        dfw, applied2, skipped2 = apply_overrides(dfw, override_map, K=K, cap=int(cap))
        # merge logs (keep moved vs already)
        applied = applied + applied2
        skipped = skipped + skipped2

    # final label
    dfw["kategori"] = dfw["_gidx"].apply(label_from_gidx)

    # build map
    m = build_map(dfw, K=K)

    return dfw, m, applied, skipped
