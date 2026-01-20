# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import streamlit.components.v1 as components

from grouping import (
    validate_input_df,
    initial_grouping,
    refine_from_current,
    apply_overrides,
    build_map,
    label_from_gidx,
    parse_label_to_gidx,
)

# -------------------------
# Helpers
# -------------------------
def df_to_excel_bytes(df: pd.DataFrame, sheet_name="grouped") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()

def template_df():
    # contoh dummy (lat/long boleh titik atau koma)
    return pd.DataFrame({
        "nama_toko": [
            "Toko Sinar Jaya", "Toko Maju Bersama", "Warung Bu Sari",
            "Toko Barokah", "Toko Aneka Rasa"
        ],
        "lat": [
            -6.20530, "-6,20610", -6.20750, "-6,20820", -6.20910
        ],
        "long": [
            106.84510, "106,84620", 106.84790, "106,84870", 106.84930
        ],
    })

def mark_need_refine():
    st.session_state.need_refine = True

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="JKS Store Grouping", layout="wide")
st.title("📍 JKS Store Grouping")

# -------------------------
# Session state init
# -------------------------
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_base" not in st.session_state:
    st.session_state.df_base = None        # hasil setelah initial grouping (+ refine pertama jika ON)
if "df_current" not in st.session_state:
    st.session_state.df_current = None     # hasil setelah override (dan refine jika dipicu)
if "override_map" not in st.session_state:
    st.session_state.override_map = {}     # {row_id: gidx}
if "need_refine" not in st.session_state:
    st.session_state.need_refine = False
if "last_file_sig" not in st.session_state:
    st.session_state.last_file_sig = None

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("⚙️ Pengaturan")

K = st.sidebar.number_input(
    "Jumlah group (R01–Rxx)",
    min_value=2, max_value=60, value=12, step=1,
    key="K",
    on_change=mark_need_refine,  # perubahan K seharusnya trigger regroup total, bukan refine saja
)

HARD_CAP = st.sidebar.number_input(
    "Max toko per group (cap)",
    min_value=5, max_value=200, value=25, step=1,
    key="HARD_CAP",
    on_change=mark_need_refine,  # parameter inti
)

st.sidebar.divider()

refine_on = st.sidebar.checkbox(
    "Refine ON (autorun saat slider digeser)",
    value=True,
    key="refine_on",
    on_change=mark_need_refine,
)

refine_iter = st.sidebar.slider(
    "Refine iterations",
    min_value=0, max_value=30, value=8, step=1,
    key="refine_iter",
    on_change=mark_need_refine,
)

neighbor_k = st.sidebar.slider(
    "Neighbor k (cari tetangga terdekat untuk swap)",
    min_value=3, max_value=60, value=12, step=1,
    key="neighbor_k",
    on_change=mark_need_refine,
)

st.sidebar.divider()
show_preview = st.sidebar.checkbox("Tampilkan preview tabel", value=True, key="show_preview")
preview_rows = st.sidebar.slider("Jumlah baris preview", 5, 500, 30, key="preview_rows")

# -------------------------
# BEFORE upload: show guide + template
# -------------------------
st.subheader("📌 Format Excel yang harus di-upload")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown(
        """
**Wajib ada kolom:**
- `nama_toko`
- `lat`
- `long`

**Format angka koordinat:**
- Boleh pakai **titik** (`-6.2053`) ✅
- Atau **koma** (`-6,2053`) ✅ (akan otomatis dikonversi)

**Catatan penting:**
- `lat` harus -90..90
- `long` harus -180..180
- Baris yang koordinatnya tidak valid akan ditolak saat validasi.
        """
    )

with col2:
    st.markdown("**Contoh data dummy:**")
    st.dataframe(template_df(), width="stretch", height=220)
    st.download_button(
        "⬇️ Download template dummy (.xlsx)",
        data=df_to_excel_bytes(template_df(), sheet_name="template"),
        file_name="template_jks_grouping.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.divider()

uploaded_file = st.file_uploader(
    "Upload Excel (.xlsx) format kolom: nama_toko | lat | long",
    type=["xlsx"]
)

if not uploaded_file:
    st.info("Upload file Excel dulu untuk mulai.")
    st.stop()

# -------------------------
# Read excel + validate
# -------------------------
try:
    df_raw = pd.read_excel(uploaded_file, sheet_name=0)
except Exception as e:
    st.error(f"Gagal baca Excel. Pastikan file .xlsx valid.\n\nDetail: {e}")
    st.stop()

if show_preview:
    st.subheader("📄 Preview Data (raw)")
    st.dataframe(df_raw.head(preview_rows), width="stretch")

ok, msg, df_clean = validate_input_df(df_raw)
if not ok:
    st.error(msg)
    st.stop()

st.success("Format file OK. Siap diproses.")

# file signature to detect changes (nama file + size)
file_sig = (getattr(uploaded_file, "name", "uploaded.xlsx"), getattr(uploaded_file, "size", None))

# kalau file berubah → reset state
if st.session_state.last_file_sig != file_sig:
    st.session_state.last_file_sig = file_sig
    st.session_state.df_clean = df_clean
    st.session_state.df_base = None
    st.session_state.df_current = None
    st.session_state.override_map = {}
    st.session_state.need_refine = True  # supaya jalan pipeline awal

# -------------------------
# Initial grouping (only once per file/parameter change)
# -------------------------
# kalau K/cap berubah, kamu lebih aman reset total (bukan refine incremental)
# karena struktur capacity constraint berubah total.
# mark_need_refine() sudah kepanggil. Di sini kita bedakan:
param_sig = (int(K), int(HARD_CAP))

if "last_param_sig" not in st.session_state:
    st.session_state.last_param_sig = None

if st.session_state.last_param_sig != param_sig:
    st.session_state.last_param_sig = param_sig
    st.session_state.df_base = None
    st.session_state.df_current = None
    st.session_state.override_map = {}
    st.session_state.need_refine = True

# Jalankan initial grouping jika belum ada
if st.session_state.df_base is None:
    with st.spinner("Initial grouping..."):
        df_base = initial_grouping(
            df_clean=st.session_state.df_clean,
            K=int(K),
            hard_cap=int(HARD_CAP),
            seed=42,
        )
        # initial_grouping() sudah ngasih gidx + kategori
        st.session_state.df_base = df_base.copy()
        st.session_state.df_current = df_base.copy()
    st.session_state.need_refine = False  # refine berikutnya hanya dari slider

# -------------------------
# Override UI
# -------------------------
st.subheader("🧷 Override (paksa pindah group secara manual)")

left, right = st.columns([1.2, 1])

with left:
    st.caption("Pilih toko lalu paksa ke group tertentu. Override tidak memicu refine otomatis.")

    df_cur = st.session_state.df_current
    # list toko (pakai id internal biar stabil)
    # grouping.py memastikan ada kolom _row_id unik
    toko_options = df_cur[["_row_id", "nama_toko", "kategori"]].copy()
    toko_options["label"] = toko_options["nama_toko"].astype(str) + "  —  " + toko_options["kategori"].astype(str)

    selected_row_id = st.selectbox(
        "Pilih toko",
        options=toko_options["_row_id"].tolist(),
        format_func=lambda rid: toko_options.loc[toko_options["_row_id"] == rid, "label"].iloc[0],
        key="override_pick_row_id",
    )

    # target group
    group_labels = [label_from_gidx(i) for i in range(int(K))]
    target_label = st.selectbox("Pindahkan ke group", options=group_labels, key="override_target_label")
    target_gidx = parse_label_to_gidx(target_label)

    colA, colB, colC = st.columns([1, 1, 1])

    with colA:
        if st.button("➕ Add / Update override", use_container_width=True):
            st.session_state.override_map[int(selected_row_id)] = int(target_gidx)
            # apply overrides tanpa refine
            st.session_state.df_current, applied, skipped = apply_overrides(
                st.session_state.df_base,
                st.session_state.override_map,
                K=int(K),
                cap=int(HARD_CAP),
            )
            st.success(f"Override applied: {applied} | skipped: {skipped}")

    with colB:
        if st.button("🗑️ Clear all overrides", use_container_width=True):
            st.session_state.override_map = {}
            st.session_state.df_current = st.session_state.df_base.copy()
            st.success("Semua override dihapus.")

    with colC:
        # tombol refine manual (opsional)
        if st.button("🔁 Refine now", use_container_width=True):
            st.session_state.need_refine = True

with right:
    st.caption("Daftar override aktif (bisa delete per baris).")
    if len(st.session_state.override_map) == 0:
        st.info("Belum ada override.")
    else:
        # tampilkan tabel override
        df_over = pd.DataFrame(
            [{"_row_id": rid, "target": label_from_gidx(g)} for rid, g in st.session_state.override_map.items()]
        )
        # join nama toko
        df_names = st.session_state.df_base[["_row_id", "nama_toko", "kategori"]].copy()
        df_over = df_over.merge(df_names, on="_row_id", how="left")
        df_over = df_over[["_row_id", "nama_toko", "kategori", "target"]]
        st.dataframe(df_over, width="stretch", height=220)

        st.caption("Delete override per toko:")
        # tombol delete per row
        for _, r in df_over.iterrows():
            rid = int(r["_row_id"])
            lbl = f"Delete: {r['nama_toko']} → {r['target']}"
            if st.button(lbl, key=f"del_override_{rid}"):
                st.session_state.override_map.pop(rid, None)
                st.session_state.df_current, applied, skipped = apply_overrides(
                    st.session_state.df_base,
                    st.session_state.override_map,
                    K=int(K),
                    cap=int(HARD_CAP),
                )
                st.success(f"Override removed. Applied: {applied} | skipped: {skipped}")
                st.rerun()

st.divider()

# -------------------------
# AUTORUN refine (ONLY if slider/toggle changed OR user clicked refine now)
# -------------------------
if st.session_state.need_refine and st.session_state.refine_on:
    with st.spinner("Refining (autorun karena parameter berubah)..."):
        # refine dari df_current (yang sudah termasuk override)
        st.session_state.df_current = refine_from_current(
            df_current=st.session_state.df_current,
            K=int(K),
            cap=int(HARD_CAP),
            refine_iter=int(st.session_state.refine_iter),
            neighbor_k=int(st.session_state.neighbor_k),
            seed=42,
            override_map=st.session_state.override_map,
        )
    st.session_state.need_refine = False

# Jika need_refine True tapi refine_off, cukup reset flag (biar gak loop)
if st.session_state.need_refine and (not st.session_state.refine_on):
    st.session_state.need_refine = False

df_result = st.session_state.df_current

# -------------------------
# Summary
# -------------------------
st.subheader("✅ Ringkasan")
st.write(f"Total titik diproses: **{len(df_result):,}**")

summary = df_result["kategori"].value_counts().sort_index().rename_axis("kategori").to_frame("jumlah")
st.dataframe(summary, width="stretch", height=260)

# -------------------------
# Map
# -------------------------
st.subheader("🗺️ Peta Grouping")
with st.spinner("Membangun peta..."):
    folium_map = build_map(df_result, show_hull=True)

components.html(folium_map._repr_html_(), height=720, scrolling=True)

# -------------------------
# Download
# -------------------------
st.subheader("⬇️ Download")
st.download_button(
    label="Download Excel hasil grouping",
    data=df_to_excel_bytes(df_result, sheet_name="grouped"),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("Lihat tabel hasil (sample 300 baris)"):
    st.dataframe(df_result.head(300), width="stretch")
