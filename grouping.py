# grouping.py
import math
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree

import folium
from folium.plugins import Search


# =========================================================
# Parsing / validation
# =========================================================
def _to_float_series(s: pd.Series) -> pd.Series:
    # support decimal comma
    s2 = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def validate_input_df(df: pd.DataFrame):
    """
    Accept headers case-insensitive, with some alias.
    Output df_clean with columns: nama_toko, lat, long
    """
    if df is None or len(df) == 0:
        return False, "File kosong.", None

    # normalize headers
    cols = {str(c).strip().lower(): c for c in df.columns}

    cand_name = ["nama_toko", "nama toko", "toko", "store", "outlet", "nama_outlet", "nama outlet", "name"]
    cand_lat = ["lat", "latitude", "y"]
    cand_lon = ["long", "lon", "longitude", "lng", "x"]

    def pick(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    c_name = pick(cand_name)
    c_lat = pick(cand_lat)
    c_lon = pick(cand_lon)

    if not c_name or not c_lat or not c_lon:
        return (
            False,
            "Kolom wajib tidak ditemukan.\n\n"
            "Wajib ada: **nama_toko**, **lat**, **long**.\n"
            "Contoh: nama_toko | lat | long",
            None,
        )

    df2 = df[[c_name, c_lat, c_lon]].copy()
    df2.columns = ["nama_toko", "lat", "long"]

    df2["nama_toko"] = df2["nama_toko"].astype(str).str.strip()
    df2["lat"] = _to_float_series(df2["lat"])
    df2["long"] = _to_float_series(df2["long"])

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
            "Ada lat/long di luar range valid.\n"
            "Lat harus -90..90, Long harus -180..180.\n\n"
            f"Contoh baris bermasalah:\n{ex.to_string(index=False)}",
            None,
        )

    return True, f"OK. Baris valid: {len(df2):,}. Baris dibuang: {bad_n:,}.", df2


# =========================================================
# Geometry: Convex Hull (Monotonic Chain) — no extra deps
# Works on (lon, lat) points in degree; good enough for visualization
# =========================================================
def _convex_hull(points):
    """
    points: list of (x, y) = (lon, lat)
    returns hull list in order, closed NOT included
    """
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

    # remove last because it's duplicated
    return lower[:-1] + upper[:-1]


# =========================================================
# Colors + labels + legend
# =========================================================
def _palette(n: int):
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#0b3d91", "#ff1493",
        "#228b22", "#ff4500", "#6a5acd", "#2f4f4f", "#b8860b", "#008080",
        "#483d8b", "#a52a2a",
    ]
    return [base[i % len(base)] for i in range(n)]


def _label(g_idx: int) -> str:
    return f"R{g_idx + 1:02d}"


def _add_legend(m: folium.Map, labels, colors):
    items = "".join(
        f"""
        <div style="display:flex;align-items:center;margin-bottom:4px;">
          <div style="width:12px;height:12px;background:{c};margin-right:8px;border:1px solid #555;"></div>
          <div style="font-size:12px;">{lab}</div>
        </div>
        """
        for lab, c in zip(labels, colors)
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
# Distance helpers (for refinement)
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


# =========================================================
# Assignment with capacity: more “local” and less lompat
# =========================================================
def _initial_kmeans(coords_deg: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init="auto", batch_size=1024)
    return km.fit_predict(coords_deg.astype(float))


def _balanced_assign_to_centroids(coords_deg: np.ndarray, centroids: np.ndarray, cap: int, soft_target: int):
    """
    Assign each point to nearest centroid with capacity.
    - soft_target ~ n/K (balanced)
    - cap is hard max
    Strategy:
    - compute all centroid distances
    - sort points by "how confident" their nearest centroid is (gap between 1st and 2nd)
    - assign to best available (prefer nearest; avoid overfilling)
    """
    n = len(coords_deg)
    K = len(centroids)

    # distances (n x K)
    d = np.linalg.norm(coords_deg[:, None, :] - centroids[None, :, :], axis=2)

    # for ordering: gap between best and second best (bigger gap = easier decision)
    part = np.partition(d, 1, axis=1)
    gap = part[:, 1] - part[:, 0]
    order = np.argsort(gap)[::-1]

    labels = -np.ones(n, dtype=int)
    counts = np.zeros(K, dtype=int)

    for i in order:
        choices = np.argsort(d[i])
        placed = False

        # pass 1: try keep cluster <= soft_target (balanced)
        for g in choices:
            if counts[g] < soft_target and counts[g] < cap:
                labels[i] = g
                counts[g] += 1
                placed = True
                break

        if placed:
            continue

        # pass 2: allow up to cap
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
# Refinement (iterative) — anti “lompat”
# =========================================================
def _refine_local_knn(
    coords_deg: np.ndarray,
    labels: np.ndarray,
    K: int,
    cap: int,
    refine_iter: int = 6,
    neighbor_k: int = 10,
    seed: int = 42,
):
    """
    Iterative local refinement:
    - Build kNN graph (haversine)
    - For each point, vote target group based on neighbor groups (weighted by inverse distance)
    - Move only if:
        - target group has room (< cap)
        - improves distance to centroid (stabilizer)
        - doesn't empty current group to zero
    Prints logs: [REFINE] iter x/y: moved=...
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
            gi = labels[i]
            if counts[gi] <= 1:
                continue

            nb_idx = nbrs[i, 1:]
            nb_dist = dist_km[i, 1:]

            # weighted vote
            scores = np.zeros(K, dtype=float)
            w = 1.0 / (nb_dist + 1e-6)
            nb_lab = labels[nb_idx]
            for lab, ww in zip(nb_lab, w):
                scores[lab] += ww

            # candidates by score
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
                new_d = np.linalg.norm(p - centroids[g])
                if new_d + 1e-9 < best_d:
                    best_g = g
                    best_d = new_d

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
# Map builder — keep “warna + hull + legend”
# =========================================================
def _build_map(df_res: pd.DataFrame, K: int):
    center = [df_res["lat"].mean(), df_res["long"].mean()]
    m = folium.Map(location=center, zoom_start=11, control_scale=True, tiles="OpenStreetMap")

    colors = _palette(K)
    labels = [_label(i) for i in range(K)]

    # add hull polygons first (so markers on top)
    for gi in range(K):
        code = _label(gi)
        sub = df_res[df_res["kategori"] == code]
        if len(sub) < 3:
            continue

        pts = list(zip(sub["long"].tolist(), sub["lat"].tolist()))  # (lon, lat)
        hull = _convex_hull(pts)
        if len(hull) < 3:
            continue

        poly_latlon = [(y, x) for (x, y) in hull]  # to (lat, lon)

        folium.Polygon(
            locations=poly_latlon,
            color=colors[gi],
            weight=2,
            fill=True,
            fill_color=colors[gi],
            fill_opacity=0.15,
            opacity=0.9,
        ).add_to(m)

    # markers
    marker_layer = folium.FeatureGroup(name="Toko", show=True).add_to(m)
    for _, r in df_res.iterrows():
        gi = int(r["_gidx"])
        folium.CircleMarker(
            location=[r["lat"], r["long"]],
            radius=5,
            color=colors[gi],
            fill=True,
            fill_color=colors[gi],
            fill_opacity=0.9,
            tooltip=f"{r['nama_toko']} ({r['kategori']})",
        ).add_to(marker_layer)

    # Search by tooltip (nama_toko)
    Search(
        layer=marker_layer,
        search_label="tooltip",
        placeholder="Cari nama toko ...",
        collapsed=False,
    ).add_to(m)

    _add_legend(m, labels, colors)
    folium.LayerControl(collapsed=True).add_to(m)

    # fit bounds
    sw = [df_res["lat"].min(), df_res["long"].min()]
    ne = [df_res["lat"].max(), df_res["long"].max()]
    m.fit_bounds([sw, ne], padding=(20, 20))

    return m


# =========================================================
# Main API
# =========================================================
def process_excel(
    df_clean: pd.DataFrame,
    hard_cap: int,
    K: int,
    refine_iter: int = 6,
    neighbor_k: int = 10,
    seed: int = 42,
):
    """
    Return:
      df_result: nama_toko, lat, long, kategori
      folium_map
    """
    dfw = df_clean.copy().reset_index(drop=True)
    n = len(dfw)

    if K < 2:
        raise ValueError("K minimal 2.")
    if hard_cap < 2:
        raise ValueError("cap minimal 2.")
    if K * hard_cap < n:
        raise ValueError(f"K*cap tidak cukup. K={K}, cap={hard_cap}, total={n}.")

    coords = dfw[["lat", "long"]].to_numpy(dtype=float)

    # balanced target (do not FORCE cap; cap only max)
    soft_target = int(math.ceil(n / K))
    soft_target = min(soft_target, int(hard_cap))

    print(f"[INFO] n={n} | K={K} | cap={hard_cap} | soft_target≈{soft_target} | refine_iter={refine_iter} | neighbor_k={neighbor_k}")

    # 1) initial clustering (kmeans centers)
    init_labels = _initial_kmeans(coords, K=K, seed=seed)
    centroids = _compute_centroids(coords, init_labels, K)

    # 2) capacity-aware balanced assignment to nearest centroids
    labels = _balanced_assign_to_centroids(coords, centroids, cap=int(hard_cap), soft_target=int(soft_target))

    # 3) iterative local refinement (anti lompat)
    labels = _refine_local_knn(
        coords_deg=coords,
        labels=labels,
        K=K,
        cap=int(hard_cap),
        refine_iter=int(refine_iter),
        neighbor_k=int(neighbor_k),
        seed=seed,
    )

    # output
    dfw["_gidx"] = labels.astype(int)
    dfw["kategori"] = dfw["_gidx"].apply(_label)

    m = _build_map(dfw, K=K)
    return dfw[["nama_toko", "lat", "long", "kategori"]].copy(), m
