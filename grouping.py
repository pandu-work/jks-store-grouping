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

    c_name = pick(["nama_toko", "nama toko", "toko", "store", "outlet", "name"])
    c_lat = pick(["lat", "latitude", "y"])
    c_lon = pick(["long", "lon", "longitude", "lng", "x"])

    if not c_name or not c_lat or not c_lon:
        return False, "Kolom wajib: nama_toko | lat | long", None

    df2 = df[[c_name, c_lat, c_lon]].copy()
    df2.columns = ["nama_toko", "lat", "long"]

    df2["nama_toko"] = df2["nama_toko"].astype(str).str.strip()
    df2["lat"] = _to_float_series(df2["lat"])
    df2["long"] = _to_float_series(df2["long"])
    df2["_row_id"] = np.arange(len(df2), dtype=int)

    bad = df2["nama_toko"].eq("") | df2["lat"].isna() | df2["long"].isna()
    bad_n = int(bad.sum())
    df2 = df2.loc[~bad].copy()

    if len(df2) == 0:
        return False, "Semua baris invalid setelah parsing lat/long.", None

    out_range = (df2["lat"].abs() > 90) | (df2["long"].abs() > 180)
    if out_range.any():
        ex = df2.loc[out_range].head(8)
        return False, f"Ada lat/long di luar range valid.\n\n{ex.to_string(index=False)}", None

    return True, f"OK. Baris valid: {len(df2):,}. Baris dibuang: {bad_n:,}.", df2


# =========================================================
# Labels / Colors
# =========================================================
def label_from_gidx(gidx: int) -> str:
    return f"R{int(gidx) + 1:02d}"


def parse_label_to_gidx(label: str) -> int:
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
      position: fixed; bottom: 18px; left: 18px; z-index: 9999;
      background: rgba(255,255,255,0.92);
      padding: 10px 12px; border-radius: 10px; border: 1px solid #ddd;
      max-height: 260px; overflow: auto;
      box-shadow: 0 4px 14px rgba(0,0,0,0.12);">
      <div style="font-weight:600;margin-bottom:8px;">Legend (Group)</div>
      {items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# =========================================================
# Convex Hull (Monotonic Chain)
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
# Core helpers
# =========================================================
def _compute_centroids(coords_deg: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    centroids = np.zeros((K, 2), dtype=float)
    for g in range(K):
        idx = np.where(labels == g)[0]
        centroids[g] = coords_deg[idx].mean(axis=0) if len(idx) else coords_deg.mean(axis=0)
    return centroids


def _initial_centroids_kmeans(coords_deg: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init="auto", batch_size=1024)
    labels0 = km.fit_predict(coords_deg.astype(float))
    return _compute_centroids(coords_deg, labels0, K)


def _balanced_assign(coords_deg: np.ndarray, centroids: np.ndarray, K: int, cap: int, soft_target: int):
    n = len(coords_deg)
    d = np.linalg.norm(coords_deg[:, None, :] - centroids[None, :, :], axis=2)

    part = np.partition(d, 1, axis=1)
    gap = part[:, 1] - part[:, 0]
    order = np.argsort(gap)[::-1]

    labels = -np.ones(n, dtype=int)
    counts = np.zeros(K, dtype=int)

    for i in order:
        choices = np.argsort(d[i])
        placed = False

        for g in choices:
            if counts[g] < soft_target and counts[g] < cap:
                labels[i] = g
                counts[g] += 1
                placed = True
                break
        if placed:
            continue

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
# Overrides (LOCK)
# =========================================================
def apply_overrides(df: pd.DataFrame, override_map: dict, K: int, cap: int):
    df = df.copy()
    df["_locked"] = False

    if not override_map:
        return df, [], []

    skipped = []
    applied = []
    counts = df["_gidx"].value_counts().to_dict()

    for rid, tgt in override_map.items():
        rid = int(rid)
        tgt = int(tgt)

        if tgt < 0 or tgt >= K:
            skipped.append((rid, "target out of range"))
            continue

        rows = df.index[df["_row_id"] == rid]
        if len(rows) == 0:
            skipped.append((rid, "row_id not found"))
            continue

        ix = rows[0]
        cur = int(df.loc[ix, "_gidx"])

        if cur == tgt:
            df.loc[ix, "_locked"] = True
            applied.append((rid, cur, tgt, "already"))
            continue

        if counts.get(tgt, 0) >= cap:
            skipped.append((rid, f"target {label_from_gidx(tgt)} full"))
            continue

        df.loc[ix, "_gidx"] = tgt
        df.loc[ix, "_locked"] = True

        counts[cur] = counts.get(cur, 0) - 1
        counts[tgt] = counts.get(tgt, 0) + 1

        applied.append((rid, cur, tgt, "moved"))

    return df, applied, skipped


# =========================================================
# Refinement (from CURRENT labels) — does NOT move locked
# =========================================================
def refine_from_current(
    df: pd.DataFrame,
    K: int,
    cap: int,
    refine_iter: int = 6,
    neighbor_k: int = 10,
    seed: int = 42,
    override_map: dict | None = None,
):
    """
    Refine based on df['_gidx'] current state.
    Locked points (df['_locked']==True) will not move.
    After refining, overrides are re-applied to ensure "override wins last".
    """
    dfw = df.copy()
    coords = dfw[["lat", "long"]].to_numpy(dtype=float)

    labels = dfw["_gidx"].to_numpy(dtype=int)
    locked_mask = dfw.get("_locked", pd.Series(False, index=dfw.index)).to_numpy(dtype=bool)

    # Build knn
    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric="haversine")
    kq = min(max(2, neighbor_k + 1), len(dfw))
    dist_rad, nbrs = tree.query(coords_rad, k=kq)
    dist_km = dist_rad * 6371.0088

    rng = np.random.default_rng(seed)
    counts = np.bincount(labels, minlength=K)

    for it in range(refine_iter):
        centroids = _compute_centroids(coords, labels, K)
        order = np.arange(len(dfw))
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
            for j, ww in zip(nb_idx, w):
                scores[int(labels[j])] += ww

            cand = np.argsort(scores)[::-1]

            p = coords[i]
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
                    best_g, best_d = g, new_d

            if best_g != gi:
                labels[i] = best_g
                counts[gi] -= 1
                counts[best_g] += 1
                moved += 1

        print(f"[REFINE] iter {it+1}/{refine_iter}: moved={moved}")
        if moved == 0:
            break

    dfw["_gidx"] = labels.astype(int)

    # reapply overrides to be safe
    if override_map:
        dfw, _, _ = apply_overrides(dfw, override_map, K=K, cap=cap)

    dfw["kategori"] = dfw["_gidx"].apply(label_from_gidx)
    return dfw


# =========================================================
# Map
# =========================================================
def build_map(df: pd.DataFrame, K: int):
    center = [df["lat"].mean(), df["long"].mean()]
    m = folium.Map(location=center, zoom_start=11, control_scale=True, tiles="OpenStreetMap")
    colors = _palette(K)

    # hull polygons
    for gi in range(K):
        sub = df[df["_gidx"] == gi]
        if len(sub) < 3:
            continue
        pts = list(zip(sub["long"].tolist(), sub["lat"].tolist()))
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

    layer = folium.FeatureGroup(name="Toko", show=True).add_to(m)

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
        ).add_to(layer)

    Search(layer=layer, search_label="tooltip", placeholder="Cari nama toko / row id ...", collapsed=False).add_to(m)
    _add_legend(m, K, colors)
    folium.LayerControl(collapsed=True).add_to(m)

    sw = [df["lat"].min(), df["long"].min()]
    ne = [df["lat"].max(), df["long"].max()]
    m.fit_bounds([sw, ne], padding=(20, 20))
    return m


# =========================================================
# Initial grouping pipeline (refine happens only here)
# =========================================================
def initial_grouping(
    df_clean: pd.DataFrame,
    K: int,
    cap: int,
    refine_on: bool = True,
    refine_iter: int = 6,
    neighbor_k: int = 10,
    seed: int = 42,
    override_map: dict | None = None,
):
    dfw = df_clean.copy().reset_index(drop=True)
    n = len(dfw)
    if K * cap < n:
        raise ValueError(f"K*cap tidak cukup. K={K}, cap={cap}, total={n}.")

    coords = dfw[["lat", "long"]].to_numpy(dtype=float)
    soft_target = min(int(math.ceil(n / K)), int(cap))

    print(f"[INFO] initial_grouping n={n} | K={K} | cap={cap} | soft_target≈{soft_target} | refine_on={refine_on}")

    centroids = _initial_centroids_kmeans(coords, K=K, seed=seed)
    labels = _balanced_assign(coords, centroids, K=K, cap=int(cap), soft_target=int(soft_target))

    dfw["_gidx"] = labels.astype(int)

    override_map = override_map or {}
    dfw, applied, skipped = apply_overrides(dfw, override_map, K=K, cap=int(cap))

    if refine_on and refine_iter > 0:
        dfw = refine_from_current(
            dfw, K=K, cap=int(cap),
            refine_iter=int(refine_iter), neighbor_k=int(neighbor_k),
            seed=seed, override_map=override_map
        )

    dfw["kategori"] = dfw["_gidx"].apply(label_from_gidx)
    return dfw, applied, skipped
