# grouping.py
import math
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, Search

# -------------------------
# Helpers: sanitize + validate
# -------------------------
REQUIRED_COLS = ["nama_toko", "lat", "long"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
        .str.lower()
    )
    return df

def validate_input_df(df_raw: pd.DataFrame):
    """
    Return: (ok: bool, message: str, df_clean: DataFrame|None)
    """
    df = normalize_columns(df_raw)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return (False,
                f"Kolom wajib tidak ada: {missing}\n\n"
                f"Kolom yang terbaca: {df.columns.tolist()}\n\n"
                "Format wajib: nama_toko | lat | long (kategori optional)",
                None)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")

    bad_num = df[df["lat"].isna() | df["long"].isna()]
    if len(bad_num) > 0:
        return (False, f"Ada {len(bad_num)} baris lat/long kosong atau bukan angka.", None)

    bad_range = df[~df["lat"].between(-90, 90) | ~df["long"].between(-180, 180)]
    if len(bad_range) > 0:
        return (False, f"Ada {len(bad_range)} baris lat/long di luar range valid.", None)

    if len(df) < 10:
        return (False, "Data terlalu sedikit (<10 baris).", None)

    df["nama_toko"] = df["nama_toko"].astype(str)

    return (True, "OK", df)

# -------------------------
# Core algo (punya kamu)
# -------------------------
EARTH_R_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_R_KM * c

def pairwise_dist(coords_latlon):
    n = coords_latlon.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i] = haversine_km(
            coords_latlon[i,0], coords_latlon[i,1],
            coords_latlon[:,0], coords_latlon[:,1]
        )
    return D

def balanced_targets(n, k, hard_cap):
    base = n // k
    rem  = n % k
    t = np.array([base + 1 if i < rem else base for i in range(k)], dtype=int)
    if t.max() > hard_cap:
        raise ValueError(f"Target rata {t.max()} > HARD_CAP={hard_cap}. Naikkan K atau HARD_CAP.")
    return t

def pick_seeds_kpp(D, k, seed=42):
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    seeds = [int(rng.integers(0, n))]
    for _ in range(1, k):
        dmin = np.min(D[:, seeds], axis=1)
        probs = (dmin ** 2)
        s = probs.sum()
        if s <= 0:
            cand = int(rng.integers(0, n))
        else:
            probs = probs / s
            cand = int(rng.choice(n, p=probs))
        seeds.append(cand)
    return seeds

def region_grow_balanced(D, k, targets, seed=42):
    n = D.shape[0]
    seeds = pick_seeds_kpp(D, k, seed=seed)

    labels = np.full(n, -1, dtype=int)
    members = [set() for _ in range(k)]
    fill = np.zeros(k, dtype=int)

    for gi, idx in enumerate(seeds):
        labels[idx] = gi
        members[gi].add(idx)
        fill[gi] += 1

    unassigned = set(range(n)) - set(seeds)

    best_dist = np.full(n, np.inf, dtype=float)
    best_group = np.full(n, -1, dtype=int)
    for i in unassigned:
        ds = [D[i, seeds[g]] for g in range(k)]
        gmin = int(np.argmin(ds))
        best_dist[i] = float(ds[gmin])
        best_group[i] = gmin

    def try_update_point(i, g, new_member):
        d = float(D[i, new_member])
        if d < best_dist[i]:
            best_dist[i] = d
            best_group[i] = g

    while unassigned:
        cand_i = None
        cand_d = np.inf

        for i in unassigned:
            g = int(best_group[i])
            if g != -1 and fill[g] < targets[g]:
                d = best_dist[i]
                if d < cand_d:
                    cand_d = d
                    cand_i = i

        if cand_i is None:
            i = next(iter(unassigned))
            best_g = None
            best_d2 = np.inf
            for g in range(k):
                if fill[g] >= targets[g]:
                    continue
                dmin = np.inf
                for m in members[g]:
                    dm = D[i, m]
                    if dm < dmin:
                        dmin = dm
                if dmin < best_d2:
                    best_d2 = dmin
                    best_g = g
            if best_g is None:
                break
            cand_i, cand_g = i, best_g
        else:
            cand_g = int(best_group[cand_i])

        labels[cand_i] = cand_g
        members[cand_g].add(cand_i)
        fill[cand_g] += 1
        unassigned.remove(cand_i)

        for i in unassigned:
            try_update_point(i, cand_g, cand_i)

    return labels

def repair_islands(D, labels, max_link_km=6.0, passes=6):
    n = len(labels)
    k = int(labels.max()) + 1

    for _ in range(passes):
        moved = False
        members = [np.where(labels == g)[0] for g in range(k)]

        for i in range(n):
            g = int(labels[i])
            if len(members[g]) <= 1:
                continue

            others = members[g][members[g] != i]
            min_self = float(np.min(D[i, others]))

            if min_self <= max_link_km:
                continue

            best_g = g
            best_d = min_self
            for gg in range(k):
                if gg == g or len(members[gg]) == 0:
                    continue
                mmin = float(np.min(D[i, members[gg]]))
                if mmin < best_d:
                    best_d = mmin
                    best_g = gg

            if best_g != g:
                labels[i] = best_g
                moved = True

        if not moved:
            break

    return labels

def majority_neighbor_reassign(D, labels, k_neighbors=8, passes=3, hard_cap=30):
    n = len(labels)
    k = int(labels.max()) + 1

    for _ in range(passes):
        counts = np.bincount(labels, minlength=k)
        changed = False

        for i in range(n):
            nn = np.argsort(D[i])[1:k_neighbors+1]
            nn_labels = labels[nn]
            vote = np.bincount(nn_labels, minlength=k)
            best = int(np.argmax(vote))

            if best != labels[i]:
                if counts[best] < hard_cap:
                    counts[labels[i]] -= 1
                    labels[i] = best
                    counts[best] += 1
                    changed = True

        if not changed:
            break

    return labels

def chain_order_centroids(centroids_latlon):
    cents = np.array(centroids_latlon)
    k = len(cents)
    overall = cents.mean(axis=0)
    start = int(np.argmin(haversine_km(np.full(k, overall[0]), np.full(k, overall[1]), cents[:,0], cents[:,1])))

    remaining = set(range(k))
    order = [start]
    remaining.remove(start)

    while remaining:
        last = order[-1]
        rem = np.array(sorted(list(remaining)))
        d = haversine_km(np.full(len(rem), cents[last,0]), np.full(len(rem), cents[last,1]), cents[rem,0], cents[rem,1])
        nxt = int(rem[np.argmin(d)])
        order.append(nxt)
        remaining.remove(nxt)
    return order

# -------------------------
# Hull + buffer (arsiran)
# -------------------------
def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convex_hull_lonlat(points_lonlat):
    pts = sorted(set(points_lonlat))
    if len(pts) <= 2:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def km_to_deg_lat(km): return km / 111.32
def km_to_deg_lon(km, lat): return km / (111.32 * max(0.2, math.cos(math.radians(lat))))

def buffer_hull(hull_lonlat, buffer_km):
    if len(hull_lonlat) < 3:
        return hull_lonlat
    lons = np.array([p[0] for p in hull_lonlat])
    lats = np.array([p[1] for p in hull_lonlat])
    c_lon = float(lons.mean())
    c_lat = float(lats.mean())
    buf_lon = km_to_deg_lon(buffer_km, c_lat)
    buf_lat = km_to_deg_lat(buffer_km)

    out = []
    for lon, lat in hull_lonlat:
        vlon = lon - c_lon
        vlat = lat - c_lat
        norm = math.sqrt(vlon*vlon + vlat*vlat) + 1e-12
        out.append((lon + (vlon/norm)*buf_lon, lat + (vlat/norm)*buf_lat))
    return out

# -------------------------
# Map builder
# -------------------------
def build_map(df, cats, name_col="nama_toko", lat_col="lat", lon_col="long", cat_col="kategori",
              boundary_buffer_km=1.2, default_zoom=11):
    palette = [
        "red","blue","green","purple","orange","darkred","lightred","beige",
        "darkblue","darkgreen","cadetblue","darkpurple","pink","lightblue",
        "lightgreen","gray","black","lightgray"
    ]
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}

    m = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()],
                   zoom_start=default_zoom, control_scale=True)

    boundary_layer = folium.FeatureGroup(name="Batas Group (Hull)", show=True)
    m.add_child(boundary_layer)

    layers = {}
    for cat in cats:
        fg = folium.FeatureGroup(name=f"{cat} (toko)", show=True)
        m.add_child(fg)
        layers[cat] = fg

    search_layer = folium.FeatureGroup(name="Cari Toko (Search)", show=True)
    m.add_child(search_layer)

    for _, r in df.iterrows():
        nm  = str(r.get(name_col, "(tanpa nama)"))
        lat = float(r[lat_col]); lon = float(r[lon_col])
        cat = str(r[cat_col])

        gmaps = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        popup = f"""
        <div style="min-width:220px">
          <div style="font-weight:700;margin-bottom:6px;">{nm}</div>
          <div style="margin-bottom:4px;">Kategori: <b>{cat}</b></div>
          <div style="margin-bottom:6px;">Lat,Lon: {lat:.6f}, {lon:.6f}</div>
          <a href="{gmaps}" target="_blank"
             style="text-decoration:none;background:#1a73e8;color:white;padding:6px 10px;border-radius:6px;display:inline-block;">
             Buka di Google Maps
          </a>
        </div>
        """
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup, max_width=340),
            icon=folium.Icon(color=color_map.get(cat, "gray"), icon="info-sign")
        ).add_to(layers.get(cat, boundary_layer))

        folium.CircleMarker([lat, lon], radius=3, fill=True, fill_opacity=0.0,
                            opacity=0.0, tooltip=nm).add_to(search_layer)

    for cat in cats:
        sub = df[df[cat_col] == cat]
        if len(sub) < 3:
            continue
        pts = list(zip(sub[lon_col].astype(float).tolist(), sub[lat_col].astype(float).tolist()))
        hull = convex_hull_lonlat(pts)
        hull = buffer_hull(hull, boundary_buffer_km)
        hull_latlon = [(lat, lon) for lon, lat in hull]
        folium.Polygon(
            locations=hull_latlon,
            color=color_map[cat],
            weight=2,
            fill=True,
            fill_opacity=0.18,
            popup=cat
        ).add_to(boundary_layer)

    HeatMap(df[[lat_col, lon_col]].dropna().values.tolist(),
            name="Heatmap", radius=18, blur=22, max_zoom=18).add_to(m)

    Search(layer=search_layer, search_label="tooltip",
           placeholder="Cari nama toko…", collapsed=False, position="topleft").add_to(m)

    bounds = [[df[lat_col].min(), df[lon_col].min()],
              [df[lat_col].max(), df[lon_col].max()]]
    m.fit_bounds(bounds)
    folium.LayerControl(collapsed=False, position="topleft").add_to(m)
    return m

# -------------------------
# MAIN FUNCTION for Streamlit
# -------------------------
def process_excel(
    df_in: pd.DataFrame,
    hard_cap: int = 25,
    K: int = 12,
    seed: int = 42,
    island_max_link_km: float = 6.0,
    island_passes: int = 8,
    k_neighbors: int = 8,
    neighbor_passes: int = 3,
    boundary_buffer_km: float = 1.2,
    default_zoom: int = 11,
):
    """
    Input  : df (sudah punya kolom nama_toko, lat, long)
    Output : (df_result, folium_map)
    """
    df = normalize_columns(df_in).copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df = df[df["lat"].between(-90, 90) & df["long"].between(-180, 180)].dropna(subset=["lat", "long"]).copy()

    coords = df[["lat", "long"]].to_numpy(dtype=float)
    n = len(coords)

    targets = balanced_targets(n, K, hard_cap)

    D = pairwise_dist(coords)

    labels = region_grow_balanced(D, K, targets, seed=seed)
    labels = repair_islands(D, labels, max_link_km=island_max_link_km, passes=island_passes)
    labels = majority_neighbor_reassign(D, labels, k_neighbors=k_neighbors, passes=neighbor_passes, hard_cap=hard_cap)

    cents = [coords[labels == g].mean(axis=0) for g in range(K)]
    order = chain_order_centroids(cents)
    remap = {old: new for new, old in enumerate(order)}

    df["kategori"] = [f"R{remap[int(l)]+1:02d}" for l in labels]
    cats = [f"R{i+1:02d}" for i in range(K)]

    m = build_map(df, cats, boundary_buffer_km=boundary_buffer_km, default_zoom=default_zoom)
    return df, m
