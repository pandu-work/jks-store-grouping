# grouping.py
import math
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree

import folium
from folium.plugins import MarkerCluster, Search

# =========================================================
# Utils
# =========================================================
def _to_float_series(s: pd.Series) -> pd.Series:
    """
    Convert series that may contain decimal commas into float.
    Examples: "107,1234" -> 107.1234
    """
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    # lat/lon degrees -> km
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _format_group_labels(labels: np.ndarray, K: int) -> pd.Series:
    # 0..K-1 -> R01..R{K}
    return pd.Series([f"R{int(x)+1:02d}" for x in labels])

def _palette(n: int):
    # nice distinct colors (repeat if needed)
    base = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
        "#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#0b3d91","#ff1493","#228b22","#ff4500","#6a5acd","#2f4f4f",
        "#b8860b","#008080","#483d8b","#a52a2a",
    ]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out

# =========================================================
# Validation
# =========================================================
def validate_input_df(df: pd.DataFrame):
    """
    Expect columns: nama_toko, lat, long
    Accept some alternatives: nama toko / latitude / longitude etc (case-insensitive)
    Handles decimal comma for lat/long.
    """
    if df is None or len(df) == 0:
        return False, "File kosong.", None

    cols = {c.lower().strip(): c for c in df.columns}

    # map possible column names
    cand_name = ["nama_toko", "nama toko", "toko", "store", "name", "nama"]
    cand_lat  = ["lat", "latitude", "y"]
    cand_lon  = ["long", "lon", "longitude", "lng", "x"]

    def pick(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    c_name = pick(cand_name)
    c_lat  = pick(cand_lat)
    c_lon  = pick(cand_lon)

    if not c_name or not c_lat or not c_lon:
        return (
            False,
            "Kolom wajib tidak ditemukan.\n\n"
            "Wajib ada: **nama_toko**, **lat**, **long**.\n"
            "Contoh header: nama_toko | lat | long",
            None
        )

    df2 = df[[c_name, c_lat, c_lon]].copy()
    df2.columns = ["nama_toko", "lat", "long"]

    # convert lat/long
    df2["lat"] = _to_float_series(df2["lat"])
    df2["long"] = _to_float_series(df2["long"])

    # drop invalid
    bad = df2["lat"].isna() | df2["long"].isna() | df2["nama_toko"].isna()
    df_bad = df2[bad]
    df2 = df2[~bad].copy()

    if len(df2) == 0:
        return False, "Semua baris invalid setelah parsing lat/long. Pastikan lat/long angka.", None

    # range check
    out_range = (df2["lat"].abs() > 90) | (df2["long"].abs() > 180)
    if out_range.any():
        ex = df2.loc[out_range].head(5)
        return (
            False,
            "Ada lat/long di luar range valid.\n"
            "Lat harus -90..90, Long harus -180..180.\n\nContoh baris bermasalah:\n"
            f"{ex.to_string(index=False)}",
            None
        )

    msg = f"OK. Baris valid: {len(df2):,}. Baris dibuang (invalid): {len(df_bad):,}."
    return True, msg, df2

# =========================================================
# Core grouping algorithm
# =========================================================
def _initial_kmeans(coords_deg: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    """
    coords_deg: Nx2 [lat, lon]
    """
    X = coords_deg.astype(float)
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init="auto", batch_size=1024)
    labels = km.fit_predict(X)
    return labels

def _compute_centroids(coords_deg: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    centroids = np.zeros((K, 2), dtype=float)
    for g in range(K):
        idx = np.where(labels == g)[0]
        if len(idx) == 0:
            centroids[g] = coords_deg.mean(axis=0)
        else:
            centroids[g] = coords_deg[idx].mean(axis=0)
    return centroids

def _enforce_cap(coords_deg: np.ndarray, labels: np.ndarray, K: int, hard_cap: int) -> np.ndarray:
    """
    If any cluster > hard_cap, push farthest-from-centroid points into other clusters with available capacity.
    """
    n = len(labels)
    if K * hard_cap < n:
        raise ValueError(f"K*cap tidak cukup. K={K}, cap={hard_cap}, total={n}. Minimal cap={math.ceil(n/K)}")

    labels = labels.copy()
    counts = np.bincount(labels, minlength=K)
    centroids = _compute_centroids(coords_deg, labels, K)

    # Build overflow list
    overflow = []
    for g in range(K):
        if counts[g] > hard_cap:
            idx = np.where(labels == g)[0]
            # distance to centroid (rough Euclidean in deg; OK for ordering)
            d = np.linalg.norm(coords_deg[idx] - centroids[g], axis=1)
            order = np.argsort(d)[::-1]  # farthest first
            move_n = counts[g] - hard_cap
            to_move = idx[order[:move_n]]
            for i in to_move:
                overflow.append(i)
            labels[to_move] = -1
            counts[g] -= move_n

    if not overflow:
        return labels

    # Assign overflow points to nearest centroid with room
    centroids = _compute_centroids(coords_deg, labels, K)  # recompute after removing
    for i in overflow:
        p = coords_deg[i]
        # candidate groups sorted by distance to centroid
        dist = np.linalg.norm(centroids - p, axis=1)
        order = np.argsort(dist)
        placed = False
        for g in order:
            if counts[g] < hard_cap:
                labels[i] = g
                counts[g] += 1
                placed = True
                break
        if not placed:
            # should not happen if K*cap >= n
            raise RuntimeError("Tidak ada group yang punya slot kosong saat enforce cap.")

    return labels

def _refine_local(
    coords_deg: np.ndarray,
    labels: np.ndarray,
    K: int,
    hard_cap: int,
    refine_iter: int = 6,
    neighbor_k: int = 10,
    seed: int = 42
) -> np.ndarray:
    """
    Iterative local refinement:
    - Precompute kNN (haversine) for each point
    - Each iter: each point votes candidate group by neighbors (weighted by 1/(dist+eps))
    - Move if it improves distance-to-centroid and target group has room (cap)
    """
    rng = np.random.default_rng(seed)
    n = len(labels)

    if refine_iter <= 0:
        return labels

    # BallTree expects radians for haversine
    coords_rad = np.radians(coords_deg)
    tree = BallTree(coords_rad, metric="haversine")

    # query k+1 because first neighbor is itself
    kq = min(max(2, neighbor_k + 1), n)
    dists, nbrs = tree.query(coords_rad, k=kq)  # distances in radians
    # convert to km
    dists_km = dists * 6371.0088

    labels = labels.copy()
    counts = np.bincount(labels, minlength=K)

    for it in range(refine_iter):
        centroids = _compute_centroids(coords_deg, labels, K)

        order = np.arange(n)
        rng.shuffle(order)

        moved = 0

        for i in order:
            gi = labels[i]

            # do not allow emptying a cluster completely (optional safety)
            if counts[gi] <= 1:
                continue

            nb_idx = nbrs[i, 1:]  # exclude itself
            nb_dist = dists_km[i, 1:]

            # neighbor group scores (weighted)
            scores = np.zeros(K, dtype=float)
            w = 1.0 / (nb_dist + 1e-6)
            nb_labels = labels[nb_idx]
            for lab, ww in zip(nb_labels, w):
                scores[lab] += ww

            # candidate groups sorted by score desc
            cand = np.argsort(scores)[::-1]

            # current distance to own centroid
            p = coords_deg[i]
            cur_d = np.linalg.norm(p - centroids[gi])

            # try best candidate(s)
            best_g = gi
            best_d = cur_d

            for g in cand[:5]:  # limit tries
                if g == gi:
                    continue
                if counts[g] >= hard_cap:
                    continue
                new_d = np.linalg.norm(p - centroids[g])
                # must improve noticeably (avoid jitter)
                if new_d + 1e-9 < best_d:
                    best_d = new_d
                    best_g = g

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
# Map builder
# =========================================================
def _build_map(df_result: pd.DataFrame, K: int):
    center_lat = df_result["lat"].mean()
    center_lon = df_result["long"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, control_scale=True, tiles="OpenStreetMap")
    colors = _palette(K)

    # layer: markers by group
    for g in range(K):
        code = f"R{g+1:02d}"
        fg = folium.FeatureGroup(name=f"{code} (toko)", show=True)
        mc = MarkerCluster(name=f"{code} cluster", disableClusteringAtZoom=14)

        sub = df_result[df_result["kategori"] == code]
        for _, r in sub.iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["long"]],
                radius=5,
                color=colors[g],
                fill=True,
                fill_opacity=0.9,
                tooltip=f"{r['nama_toko']} ({code})",
                popup=folium.Popup(
                    f"<b>{r['nama_toko']}</b><br>"
                    f"Group: {code}<br>"
                    f"Index: {int(r['_idx'])}<br>",
                    max_width=300
                ),
            ).add_to(mc)

        mc.add_to(fg)
        fg.add_to(m)

    # Search by nama_toko (use hidden layer)
    search_fg = folium.FeatureGroup(name="Cari Toko (Search)", show=True)
    for _, r in df_result.iterrows():
        folium.Marker(
            location=[r["lat"], r["long"]],
            tooltip=r["nama_toko"],
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(search_fg)
    search_fg.add_to(m)

    Search(
        layer=search_fg,
        search_label="tooltip",
        placeholder="Cari nama toko ...",
        collapsed=False,
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
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
    seed: int = 42
):
    """
    Returns: (df_result, folium_map)
    df_result includes:
      - nama_toko, lat, long, kategori
    """

    dfw = df_clean.copy().reset_index(drop=True)
    dfw["_idx"] = np.arange(len(dfw))

    coords = dfw[["lat", "long"]].to_numpy(dtype=float)
    n = len(coords)

    if K < 2:
        raise ValueError("K minimal 2.")
    if hard_cap < 2:
        raise ValueError("cap minimal 2.")
    if K * hard_cap < n:
        raise ValueError(f"K*cap tidak cukup. K={K}, cap={hard_cap}, total={n}.")

    print(f"[INFO] n={n} | K={K} | cap={hard_cap} | refine_iter={refine_iter} | neighbor_k={neighbor_k}")

    # 1) initial clustering
    labels = _initial_kmeans(coords, K=K, seed=seed)

    # 2) enforce cap
    labels = _enforce_cap(coords, labels, K=K, hard_cap=hard_cap)

    # 3) iterative local refinement (the key to reduce "lompat")
    labels = _refine_local(
        coords_deg=coords,
        labels=labels,
        K=K,
        hard_cap=hard_cap,
        refine_iter=int(refine_iter),
        neighbor_k=int(neighbor_k),
        seed=seed
    )

    # final enforce (safety)
    labels = _enforce_cap(coords, labels, K=K, hard_cap=hard_cap)

    dfw["kategori"] = _format_group_labels(labels, K=K).values

    # build map
    m = _build_map(dfw, K=K)

    return dfw[["nama_toko", "lat", "long", "kategori", "_idx"]].copy(), m
