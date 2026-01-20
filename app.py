# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import streamlit.components.v1 as components

from grouping import (
    validate_input_df,
    run_grouping_pipeline,
    label_from_gidx,
    parse_label_to_gidx,
)

st.set_page_config(page_title="JKS Store Grouping", layout="wide")
st.title("📍 JKS Store Grouping")

# =========================================================
# Helpers
# =========================================================
def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "grouped") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()


def file_fingerprint(uploaded_file) -> str:
    # simple, good enough: name + size
    return f"{uploaded_file.name}::{uploaded_file.size}"


def make_dummy_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nama_toko": ["TOKO ALFA", "TOKO BETA", "TOKO GAMMA"],
            "lat": [-6.2001, -6.2015, -6.1989],
            "long": [106.8167, 106.8201, 106.8122],
        }
    )


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("⚙️ Pengaturan")
K = st.sidebar.number_input("Jumlah group (R01–Rxx)", min_value=2, max_value=30, value=12, step=1)
CAP = st.sidebar.number_input("Max toko per group (cap)", min_value=5, max_value=50, value=25, step=1)

st.sidebar.divider()
st.sidebar.subheader("🧠 Refinement (anti lompat)")
REFINE_ON = st.sidebar.toggle("Refine ON", value=True)
REFINE_ITER = st.sidebar.slider("Iterasi refine", 0, 20, 6)
NEIGHBOR_K = st.sidebar.slider("Tetangga (k-NN)", 3, 30, 10)

st.sidebar.divider()
st.sidebar.subheader("✋ Manual Override")
st.sidebar.caption("Pilih toko via Row ID (index Excel). Titik override akan 🔒 (locked).")

# =========================================================
# Landing: format + dummy + template download
# =========================================================
st.markdown("### 📌 Format Excel yang harus di-upload")
st.markdown(
    """
Kolom wajib (case-insensitive):
- **nama_toko**
- **lat**
- **long**

Decimal separator:
- ✅ `-6.2001` (titik)
- ✅ `-6,2001` (koma) — akan otomatis dikonversi
"""
)

dummy = make_dummy_template()
c1, c2 = st.columns([2, 1], vertical_alignment="top")
with c1:
    st.caption("Contoh data dummy:")
    st.dataframe(dummy, use_container_width=True)
with c2:
    st.caption("Template:")
    st.download_button(
        "⬇️ Download Template Excel",
        data=df_to_excel_bytes(dummy, sheet_name="template"),
        file_name="template_jks_store_grouping.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.divider()

# =========================================================
# Upload
# =========================================================
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload file Excel dulu untuk mulai.")
    st.stop()

fp = file_fingerprint(uploaded)

# =========================================================
# Reset state if file changed
# =========================================================
if "file_fp" not in st.session_state or st.session_state.file_fp != fp:
    st.session_state.file_fp = fp
    st.session_state.override_map = {}  # {_row_id: gidx_target}
    st.session_state.last_K = int(K)
    st.session_state.last_CAP = int(CAP)
    st.session_state.notice = "File baru terdeteksi → override direset."

# Reset overrides if K changed (structure change)
if "last_K" in st.session_state and int(K) != int(st.session_state.last_K):
    st.session_state.override_map = {}
    st.session_state.notice = f"K berubah {st.session_state.last_K} → {int(K)} → override direset."
    st.session_state.last_K = int(K)

# If CAP changed, keep overrides but warn if some become impossible later
if "last_CAP" in st.session_state and int(CAP) != int(st.session_state.last_CAP):
    st.session_state.notice = f"Cap berubah {st.session_state.last_CAP} → {int(CAP)}. Override tetap disimpan, tapi bisa ada yang gagal jika target group penuh."
    st.session_state.last_CAP = int(CAP)

if "notice" in st.session_state and st.session_state.notice:
    st.info(st.session_state.notice)
    st.session_state.notice = ""

# =========================================================
# Read + validate
# =========================================================
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

# =========================================================
# Run pipeline (grouping + override + refine)
# =========================================================
override_map = st.session_state.get("override_map", {})

with st.spinner("Processing grouping & membuat peta..."):
    try:
        df_result, folium_map, applied, skipped = run_grouping_pipeline(
            df_clean=df_clean,
            K=int(K),
            cap=int(CAP),
            override_map=override_map,
            refine_on=bool(REFINE_ON),
            refine_iter=int(REFINE_ITER),
            neighbor_k=int(NEIGHBOR_K),
        )
    except Exception as e:
        st.error(f"Proses gagal.\n\nDetail: {e}")
        st.stop()

# =========================================================
# Manual Override UI (needs df_result to list options)
# =========================================================
# Build options like "[0007] TOKO ABC (R03)" for clarity
options = [
    (int(r["_row_id"]), f"[{int(r['_row_id']):04d}] {r['nama_toko']} ({r['kategori']})")
    for _, r in df_result[["_row_id", "nama_toko", "kategori"]].iterrows()
]

# Keep a stable mapping row_id -> label string for selectbox
opt_labels = [x[1] for x in options]
opt_rowids = [x[0] for x in options]

if len(opt_labels) > 0:
    selected_label = st.sidebar.selectbox("Pilih toko (Row ID)", opt_labels, index=0)
    selected_row_id = opt_rowids[opt_labels.index(selected_label)]

    target_label = st.sidebar.selectbox(
        "Pindahkan ke group",
        [label_from_gidx(i) for i in range(int(K))]
    )

    # show current assignment + capacity quick view
    cur_row = df_result[df_result["_row_id"] == selected_row_id].iloc[0]
    st.sidebar.caption(f"Current: **{cur_row['kategori']}** | Locked: **{'Yes' if cur_row.get('_locked', False) else 'No'}**")

    # capacity summary
    vc = df_result["kategori"].value_counts().sort_index()
    st.sidebar.caption("Capacity:")
    for g in range(int(K)):
        lab = label_from_gidx(g)
        st.sidebar.write(f"- {lab}: {int(vc.get(lab, 0))}/{int(CAP)}")

    colA, colB = st.sidebar.columns(2)
    with colA:
        if st.button("Apply Override", type="primary"):
            gidx_target = parse_label_to_gidx(target_label)
            st.session_state.override_map[selected_row_id] = int(gidx_target)
            st.session_state.notice = f"Override set: Row {selected_row_id} → {target_label}"
            st.rerun()

    with colB:
        if st.button("Remove Override"):
            if selected_row_id in st.session_state.override_map:
                del st.session_state.override_map[selected_row_id]
                st.session_state.notice = f"Override removed: Row {selected_row_id}"
                st.rerun()

    if st.sidebar.button("Reset ALL Overrides"):
        st.session_state.override_map = {}
        st.session_state.notice = "Semua override direset."
        st.rerun()

# =========================================================
# Show override logs
# =========================================================
with st.expander("🧾 Log Override & Warnings", expanded=False):
    st.write("Applied overrides (row_id, from, to, status):")
    if applied:
        st.dataframe(pd.DataFrame(applied, columns=["row_id", "from_gidx", "to_gidx", "status"]))
    else:
        st.caption("No applied overrides in this run.")

    st.write("Skipped overrides:")
    if skipped:
        st.dataframe(pd.DataFrame(skipped, columns=["row_id", "reason"]))
    else:
        st.caption("No skipped overrides.")

# =========================================================
# Summary + Map
# =========================================================
st.subheader("✅ Ringkasan")
st.write(f"Total titik diproses: **{len(df_result):,}**")
st.dataframe(
    df_result["kategori"].value_counts().sort_index().rename_axis("kategori").to_frame("jumlah"),
    use_container_width=True
)

st.subheader("🗺️ Peta Grouping")
components.html(folium_map._repr_html_(), height=720, scrolling=True)

# =========================================================
# Download (ensures it reflects overrides)
# =========================================================
st.subheader("⬇️ Download")
export_cols = ["_row_id", "nama_toko", "lat", "long", "kategori"]
st.download_button(
    "Download Excel hasil grouping (includes overrides)",
    data=df_to_excel_bytes(df_result[export_cols].copy(), sheet_name="grouped"),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("Lihat tabel hasil (sample 200 baris)"):
    st.dataframe(df_result[export_cols].head(200), use_container_width=True)
