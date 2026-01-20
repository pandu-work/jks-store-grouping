# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import streamlit.components.v1 as components

from grouping import (
    validate_input_df,
    initial_grouping,
    apply_overrides,
    refine_from_current,
    build_map,
    label_from_gidx,
    parse_label_to_gidx,
)

st.set_page_config(page_title="JKS Store Grouping", layout="wide")
st.title("📍 JKS Store Grouping")

# -------------------------
# Helpers
# -------------------------
def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "grouped") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()

def file_fingerprint(uploaded_file) -> str:
    return f"{uploaded_file.name}::{uploaded_file.size}"

def dummy_template():
    return pd.DataFrame({
        "nama_toko": ["TOKO ALFA", "TOKO BETA", "TOKO GAMMA"],
        "lat": [-6.2001, -6.2015, -6.1989],
        "long": [106.8167, 106.8201, 106.8122],
    })

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("⚙️ Pengaturan")
K = st.sidebar.number_input("Jumlah group (R01–Rxx)", 2, 30, 12, 1)
CAP = st.sidebar.number_input("Max toko per group (cap)", 5, 50, 25, 1)

st.sidebar.divider()
st.sidebar.subheader("🧠 Refinement (awal saja)")
REFINE_ON_INIT = st.sidebar.toggle("Refine saat initial run", value=True)
REFINE_ITER = st.sidebar.slider("Iterasi refine", 0, 20, 6)
NEIGHBOR_K = st.sidebar.slider("Tetangga (k-NN)", 3, 30, 10)

st.sidebar.divider()
st.sidebar.subheader("✋ Manual Override")
st.sidebar.caption("Override tidak akan memicu refine ulang. Refine ulang hanya lewat tombol khusus.")

# -------------------------
# Landing
# -------------------------
st.markdown("### 📌 Format Excel yang harus di-upload")
st.markdown("- kolom wajib: **nama_toko**, **lat**, **long** (koma/titik untuk desimal boleh)")
tmp = dummy_template()
c1, c2 = st.columns([2, 1], vertical_alignment="top")
with c1:
    st.caption("Contoh dummy:")
    st.dataframe(tmp, use_container_width=True)
with c2:
    st.caption("Template:")
    st.download_button(
        "⬇️ Download Template Excel",
        data=df_to_excel_bytes(tmp, sheet_name="template"),
        file_name="template_jks_store_grouping.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.divider()

# -------------------------
# Upload
# -------------------------
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.stop()

fp = file_fingerprint(uploaded)

# -------------------------
# Reset state on file / K changes
# -------------------------
if "file_fp" not in st.session_state or st.session_state.file_fp != fp:
    st.session_state.file_fp = fp
    st.session_state.override_map = {}   # {_row_id: gidx_target}
    st.session_state.df_base = None      # base grouped df (after initial)
    st.session_state.last_K = int(K)
    st.session_state.last_CAP = int(CAP)
    st.session_state.notice = "File baru → state direset."

if int(K) != int(st.session_state.get("last_K", K)):
    st.session_state.override_map = {}
    st.session_state.df_base = None
    st.session_state.notice = f"K berubah {st.session_state.last_K} → {int(K)} → regroup + override reset."
    st.session_state.last_K = int(K)

# If cap changes, we keep base but we should re-check overrides on next apply (can become full)
if int(CAP) != int(st.session_state.get("last_CAP", CAP)):
    st.session_state.notice = f"Cap berubah {st.session_state.last_CAP} → {int(CAP)}. Override tetap ada, tapi bisa gagal kalau target penuh."
    st.session_state.last_CAP = int(CAP)

if st.session_state.get("notice"):
    st.info(st.session_state.notice)
    st.session_state.notice = ""

# -------------------------
# Read + validate
# -------------------------
try:
    df_raw = pd.read_excel(uploaded, sheet_name=0)
except Exception as e:
    st.error(f"Gagal baca Excel.\n\nDetail: {e}")
    st.stop()

ok, msg, df_clean = validate_input_df(df_raw)
if not ok:
    st.error(msg)
    st.stop()
st.success(msg)

# -------------------------
# Initial grouping (ONLY once per file/K)
# -------------------------
if st.session_state.df_base is None:
    with st.spinner("Initial grouping (dan refine jika ON)..."):
        df_base, _, _ = initial_grouping(
            df_clean=df_clean,
            K=int(K),
            cap=int(CAP),
            refine_on=bool(REFINE_ON_INIT),
            refine_iter=int(REFINE_ITER),
            neighbor_k=int(NEIGHBOR_K),
            override_map=st.session_state.override_map,
        )
    st.session_state.df_base = df_base

# df_work = base + overrides applied (NO refine on override)
df_work = st.session_state.df_base.copy()

# apply overrides (locked) on top of current base
df_work, applied, skipped = apply_overrides(
    df_work,
    st.session_state.override_map,
    K=int(K),
    cap=int(CAP),
)

# recalc kategori (apply_overrides sets _gidx but we ensure label)
df_work["kategori"] = df_work["_gidx"].apply(label_from_gidx)

# -------------------------
# Sidebar override UI
# -------------------------
options = [
    (int(r["_row_id"]), f"[{int(r['_row_id']):04d}] {r['nama_toko']} ({r['kategori']})")
    for _, r in df_work[["_row_id", "nama_toko", "kategori"]].iterrows()
]
opt_labels = [x[1] for x in options]
opt_rowids = [x[0] for x in options]

if opt_labels:
    selected_label = st.sidebar.selectbox("Pilih toko (Row ID)", opt_labels, index=0)
    selected_row_id = opt_rowids[opt_labels.index(selected_label)]
    target_label = st.sidebar.selectbox("Pindahkan ke group", [label_from_gidx(i) for i in range(int(K))])

    colA, colB = st.sidebar.columns(2)
    with colA:
        if st.sidebar.button("Apply Override", type="primary"):
            st.session_state.override_map[selected_row_id] = int(parse_label_to_gidx(target_label))
            st.rerun()
    with colB:
        if st.sidebar.button("Remove Override"):
            if selected_row_id in st.session_state.override_map:
                del st.session_state.override_map[selected_row_id]
                st.rerun()

    if st.sidebar.button("Reset ALL Overrides"):
        st.session_state.override_map = {}
        st.rerun()

st.sidebar.divider()

# -------------------------
# Optional: Re-Refine button (explicit)
# -------------------------
if st.sidebar.button("🔁 Re-Refine Now (manual)"):
    with st.spinner("Refining ulang dari state sekarang..."):
        # IMPORTANT: refine from current base WITH overrides locked
        df_tmp = df_work.copy()
        df_tmp = refine_from_current(
            df_tmp,
            K=int(K),
            cap=int(CAP),
            refine_iter=int(REFINE_ITER),
            neighbor_k=int(NEIGHBOR_K),
            override_map=st.session_state.override_map,
        )
        # after re-refine, update base (so next override does not trigger refine again)
        st.session_state.df_base = df_tmp.copy()
    st.rerun()

# -------------------------
# Build map from df_work (NO auto-refine on override)
# -------------------------
folium_map = build_map(df_work, K=int(K))

# -------------------------
# Summary + Map
# -------------------------
st.subheader("✅ Ringkasan")
st.write(f"Total titik diproses: **{len(df_work):,}**")
st.dataframe(
    df_work["kategori"].value_counts().sort_index().rename_axis("kategori").to_frame("jumlah"),
    use_container_width=True
)

components.html(folium_map._repr_html_(), height=720, scrolling=True)

# -------------------------
# Override logs
# -------------------------
with st.expander("🧾 Override Log", expanded=False):
    if applied:
        st.write("Applied (row_id, from, to, status):")
        st.dataframe(pd.DataFrame(applied, columns=["row_id", "from_gidx", "to_gidx", "status"]))
    else:
        st.caption("No applied overrides.")

    if skipped:
        st.write("Skipped (row_id, reason):")
        st.dataframe(pd.DataFrame(skipped, columns=["row_id", "reason"]))
    else:
        st.caption("No skipped overrides.")

# -------------------------
# Download (reflect current overrides)
# -------------------------
st.subheader("⬇️ Download")
export_cols = ["_row_id", "nama_toko", "lat", "long", "kategori"]
st.download_button(
    "Download Excel hasil (mengikuti override saat ini)",
    data=df_to_excel_bytes(df_work[export_cols].copy(), sheet_name="grouped"),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("Lihat tabel hasil (sample 200 baris)"):
    st.dataframe(df_work[export_cols].head(200), use_container_width=True)
