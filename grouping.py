# grouping.py
import math
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree

import folium
from folium.plugins import Search

# =========================================================
# Helpers: parsing
# =========================================================
def _to_float_series(s: pd.Series) -> pd.Series:
    # allow comma decimal
    s2 = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")

def label_from_gidx(gidx: int) -> str:
    return f"R{int(gidx)+1:02d}"

def parse_label_to_gidx(label: str) -> int:
    # label "R01" -> 0
    label = str(label).strip().upper()
    if label.startswith("R"):
        label = label[1:]
    return int(label) - 1

# =========================================================
# Validation
# =========================================================
def validate_input_df(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return False, "File kosong.", None

    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    c_name = pick(["nama_toko", "nama", "toko", "store", "name"])
    c_lat = pick(["lat", "latitude"])
    c_lon = pick(["long", "lon", "lng", "longitude"])

    if c_name is None or c_lat is None or c_lon is None:
        return (
            False,
            "Kolom wajib tidak lengkap. Wajib ada kolom: nama_toko, lat, long.\n"
            f"Kolom terdeteksi: {list(df.columns)}",
            None,
        )

    out = df[[c_name, c_lat, c_lon]].copy()
    out.columns = ["nama_toko", "lat", "long"]

    out["lat"] = _to_float_series(out["lat"])
    out["long"] = _to_float_series(out["long"])

    bad = out["lat"].isna() | out["long"].isna()
    if bad.any():
        n = int(bad.sum())
        return False, f"Ada {n} baris yang lat/long tidak bisa dibaca (kosong / format salah).", None

    if not ((out["lat"].between(-90, 90)) & (out["long"].between(-180, 180))).all():
        return False, "Ada lat/long di luar range valid (lat -90..90, long -180..180).", None

    out = out.reset_index(drop=True)
    out["_row_id"] = np.arange(len(out), dtype=int)  # id stabil untuk override
    return True, "OK", out

# =========================================================
# Core: balanced assignment with capacity
# =========================================================
def _compute_centroids(coords_xy: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    centroids = np.zeros((K, 2), dtype=float)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            centroids[k] = coords_xy.mean(axis=0)
        else:
            centroids[k] = coords_xy[idx].mean(axis=0)
    return centroids

def _balanced_assign(coords_xy: np.ndarray, centroids: np.ndarray, K: int, cap: int, seed: int = 42) -> np.ndarray:
    """
    Greedy assignment:
    - order points by "how certain" (difference between nearest & 2nd nearest centroid)
    - assign to nearest centroid with available capacity; if full, try next nearest.
    """
    rng = np.random.default_rng(seed)

    # distances N x K
    d2 = ((coords_xy[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argsort(d2, axis=1)

    # certainty = gap between best and second best
    best = d2[np.arange(len(coords_xy)), nearest[:, 0]]
    second = d2[np.arange(len(coords_xy)), nearest[:, 1]]
    certainty = second - best

    order = np.argsort(-certainty)  # most certain first

    labels = np.full(len(coords_xy), -1, dtype=int)
    counts = np.zeros(K, dtype=int)

    for i in order:
        for k in nearest[i]:
            if counts[k] < cap:
                labels[i] = k
                counts[k] += 1
                break

        if labels[i] == -1:
            # all full -> place to smallest count (should be rare if cap feasible)
            k = int(np.argmin(counts))
            labels[i] = k
            counts[k] += 1

    return labels

def _initial_centroids_kmeans(coords_xy: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=2048, n_init="auto")
    km.fit(coords_xy)
    return km.cluster_centers_.astype(float)

# =========================================================
# Refinement: local swap to reduce within-group distance
# =========================================================
def refine_from_current(
    df_current: pd.DataFrame,
    K: int,
    cap: int,
    refine_iter: int = 8,
    neighbor_k: int = 12,
    seed: int = 42,
    override_map: dict | None = None,
) -> pd.DataFrame:
    """
    Local refinement:
    - For each point, look at nearest neighbors
    - Try moving point to neighbor's cluster if it improves objective
    - Keep capacity constraint
    - Keep overrides locked (row_id in override_map cannot move away from target)
    """
    if refine_iter <= 0:
        out = df_current.copy()
        out["kategori"] = out["_gidx"].apply(label_from_gidx)
        return out

    dfw = df_current.copy()

    override_map = override_map or {}
    locked = set(int(k) for k in override_map.keys())

    coords = dfw[["lat", "long"]].to_numpy(dtype=float)
    # use (lon,lat) planar approx for distance
    coords_xy = np.column_stack([dfw["long"].to_numpy(float), dfw["lat"].to_numpy(float)])

    labels = dfw["_gidx"].to_numpy(dtype=int)

    # BallTree for neighbors
    tree = BallTree(coords_xy, metric="euclidean")
    nn_d, nn_i = tree.query(coords_xy, k=min(neighbor_k + 1, len(dfw)))

    rng = np.random.default_rng(seed)

    def objective(labels_local):
        cents = _compute_centroids(coords_xy, labels_local, K)
        d = np.sqrt(((coords_xy - cents[labels_local]) ** 2).sum(axis=1))
        return float(d.mean())

    # precompute counts
    counts = np.bincount(labels, minlength=K)

    base_obj = objective(labels)

    for it in range(int(refine_iter)):
        order = rng.permutation(len(dfw))

        cents = _compute_centroids(coords_xy, labels, K)

        improved = False

        for idx in order:
            row_id = int(dfw.loc[idx, "_row_id"])
            if row_id in locked:
                # stay on forced target
                forced = int(override_map[row_id])
                if labels[idx] != forced:
                    # fix if somehow drifted
                    old = labels[idx]
                    if counts[forced] < cap:
                        labels[idx] = forced
                        counts[old] -= 1
                        counts[forced] += 1
                continue

            cur = int(labels[idx])
            # try candidate clusters from neighbors
            neighs = nn_i[idx][1:]  # exclude self
            cand_clusters = list(dict.fromkeys(labels[neighs].tolist()))  # unique preserve order

            # also try closest centroid cluster
            # (helps jump between neighbor sets)
            d2 = ((coords_xy[idx] - cents) ** 2).sum(axis=1)
            cand_clusters = list(dict.fromkeys([int(np.argmin(d2))] + cand_clusters))

            best_k = cur
            best_gain = 0.0

            # current distance to centroid
            cur_dist = np.sqrt(((coords_xy[idx] - cents[cur]) ** 2).sum())

            for k in cand_clusters:
                k = int(k)
                if k == cur:
                    continue
                if counts[k] >= cap:
                    continue

                new_dist = np.sqrt(((coords_xy[idx] - cents[k]) ** 2).sum())
                gain = cur_dist - new_dist  # positive means improvement

                if gain > best_gain:
                    best_gain = gain
                    best_k = k

            if best_k != cur and best_gain > 1e-9:
                labels[idx] = best_k
                counts[cur] -= 1
                counts[best_k] += 1
                improved = True

        new_obj = objective(labels)
        if (not improved) or (new_obj >= base_obj - 1e-12):
            break
        base_obj = new_obj

    dfw["_gidx"] = labels.astype(int)
    dfw["kategori"] = dfw["_gidx"].apply(label_from_gidx)
    return dfw

# =========================================================
# Apply overrides (no refine)
# =========================================================
def apply_overrides(df_base: pd.DataFrame, override_map: dict, K: int, cap: int):
    """
    Apply overrides on top of df_base WITHOUT refining.
    If target group full -> skip that override (report skipped).
    """
    dfw = df_base.copy()
    labels = dfw["_gidx"].to_numpy(dtype=int)

    counts = np.bincount(labels, minlength=K)

    applied = 0
    skipped = 0

    # apply in deterministic order
    for row_id, target in sorted(override_map.items(), key=lambda x: int(x[0])):
        row_id = int(row_id)
        target = int(target)

        idxs = np.where(dfw["_row_id"].to_numpy(dtype=int) == row_id)[0]
        if len(idxs) == 0:
            skipped += 1
            continue

        idx = int(idxs[0])
        cur = int(labels[idx])

        if target < 0 or target >= K:
            skipped += 1
            continue

        if cur == target:
            applied += 1
            continue

        if counts[target] >= cap:
            skipped += 1
            continue

        # move
        labels[idx] = target
        counts[cur] -= 1
        counts[target] += 1
        applied += 1

    dfw["_gidx"] = labels.astype(int)
    dfw["kategori"] = dfw["_gidx"].apply(label_from_gidx)
    return dfw, applied, skipped

# =========================================================
# Initial grouping
# =========================================================
def initial_grouping(df_clean: pd.DataFrame, K: int, hard_cap: int, seed: int = 42) -> pd.DataFrame:
    """
    Initial grouping:
    - kmeans centroids (lon,lat planar)
    - balanced assignment with cap
    """
    dfw = df_clean.copy()

    coords_xy = np.column_stack([dfw["long"].to_numpy(float), dfw["lat"].to_numpy(float)])

    # feasibility quick check
    if len(dfw) > K * hard_cap:
        raise ValueError(
            f"Data {len(dfw)} baris > K*cap = {K}*{hard_cap} = {K*hard_cap}. "
            f"Naikkan cap atau naikkan K."
        )

    centroids = _initial_centroids_kmeans(coords_xy, K=K, seed=seed)
    labels = _balanced_assign(coords_xy, centroids, K=K, cap=hard_cap, seed=seed)

    dfw["_gidx"] = labels.astype(int)
    dfw["kategori"] = dfw["_gidx"].apply(label_from_gidx)
    return dfw

# =========================================================
# Folium map building (colors + hull + popup gmaps + search)
# =========================================================
def _palette(n: int):
    base = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2",
        "#7f7f7f","#bcbd22","#17becf","#0b84a5","#f6c85f","#6f4e7c","#9dd866",
        "#ca472f","#ffa056","#8dddd0","#b07aa1","#59a14f","#edc948"
    ]
    if n <= len(base):
        return base[:n]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out

def _convex_hull(points_xy: np.ndarray):
    """
    Monotonic chain convex hull (planar).
    points_xy: (N,2) lon/lat
    returns hull points in order
    """
    pts = np.array(points_xy, dtype=float)
    if len(pts) < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull

def build_map(df_result: pd.DataFrame, show_hull: bool = True) -> folium.Map:
    dfw = df_result.copy()

    center_lat = float(dfw["lat"].mean())
    center_lon = float(dfw["long"].mean())

    K = int(dfw["_gidx"].max()) + 1 if len(dfw) else 1
    colors = _palette(K)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap", control_scale=True)

    # layer group per kategori
    layers = {}
    for k in range(K):
        lab = label_from_gidx(k)
        layers[lab] = folium.FeatureGroup(name=f"{lab} (toko)", show=True)

    # add markers
    for _, r in dfw.iterrows():
        name = str(r["nama_toko"])
        lat = float(r["lat"])
        lon = float(r["long"])
        gidx = int(r["_gidx"])
        lab = label_from_gidx(gidx)

        gmaps = f"https://www.google.com/maps?q={lat},{lon}"
        html = f"""
        <div style="font-size:13px">
          <b>{name}</b><br/>
          Group: <b>{lab}</b><br/>
          Lat/Long: {lat}, {lon}<br/>
          <a href="{gmaps}" target="_blank">📍 Go to Google Maps</a>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=colors[gidx],
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(html, max_width=320),
            tooltip=f"{name} — {lab}",
        ).add_to(layers[lab])

    # add to map
    for lab, fg in layers.items():
        fg.add_to(m)

    # Hull overlay
    if show_hull:
        hull_layer = folium.FeatureGroup(name="Batas Group (Hull)", show=True)
        for k in range(K):
            sub = dfw[dfw["_gidx"] == k]
            if len(sub) < 3:
                continue
            pts = np.column_stack([sub["long"].to_numpy(float), sub["lat"].to_numpy(float)])
            hull = _convex_hull(pts)
            # hull coords -> latlon
            hull_latlon = [(float(p[1]), float(p[0])) for p in hull]
            folium.Polygon(
                locations=hull_latlon,
                color=colors[k],
                weight=2,
                fill=True,
                fill_opacity=0.12,
            ).add_to(hull_layer)
        hull_layer.add_to(m)

    # Search by tooltip (nama + group)
    # Use a hidden GeoJson points for search
    features = []
    for _, r in dfw.iterrows():
        features.append({
            "type": "Feature",
            "properties": {"name": str(r["nama_toko"])},
            "geometry": {"type": "Point", "coordinates": [float(r["long"]), float(r["lat"])]}
        })

    gj = folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="Cari Toko (Search)",
        show=True,
        tooltip=folium.GeoJsonTooltip(fields=["name"]),
    )
    gj.add_to(m)

    Search(
        layer=gj,
        geom_type="Point",
        search_label="name",
        placeholder="Cari nama toko ...",
        collapsed=False,
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

# =========================================================
# Compatibility wrapper (old style)
# =========================================================
def process_excel(df_clean: pd.DataFrame, hard_cap: int, K: int):
    """
    Legacy wrapper: return df_result, folium_map
    """
    df_base = initial_grouping(df_clean, K=K, hard_cap=hard_cap, seed=42)
    # by default no refine here, caller can refine
    fmap = build_map(df_base, show_hull=True)
    return df_base, fmap
