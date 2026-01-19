# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import streamlit.components.v1 as components

from grouping import validate_input_df, process_excel


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="JKS Store Grouping", layout="wide")
SHOW_DEBUG = False  # ubah True kalau mau lihat path file & info debugging


# =========================
# HELPERS
# =========================
def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "grouped") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()


def render_template_section():
    st.markdown("### 📌 Format file Excel yang harus di-upload")

    st.markdown(
        """
**Kolom WAJIB (case-insensitive):**
- `nama_toko` → nama outlet
- `lat` → latitude (decimal degree)
- `long` → longitude (decimal degree)

**Catatan penting:**
- Koordinat idealnya pakai **titik (.)** untuk desimal (contoh `-6.3064`)
- Kalau file kamu pakai **koma (,)**, sistem akan coba **otomatis mengonversi** (contoh `-6,3064`)
"""
    )

    st.code("nama_toko | lat | long", language="text")

    df_dummy = pd.DataFrame(
        {
            "nama_toko": ["Toko Sumber Rejeki", "Warung Bu Nita", "Kios Asep Jaya"],
            "lat": [-6.306400, -6.295120, -6.315880],
            "long": [107.149558, 107.162330, 107.141220],
        }
    )

    st.markdown("#### Contoh data (dummy):")
    st.dataframe(df_dummy, use_container_width=True)

    st.download_button(
        label="⬇️ Download template Excel",
        data=df_to_excel_bytes(df_dummy, sheet_name="template"),
        file_name="template_jks_store_grouping.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =========================
# UI HEADER
# =========================
st.title("📍 JKS Store Grouping")
st.caption("Upload Excel → sistem membagi toko menjadi beberapa group yang berdekatan + peta hasil grouping.")

# Sidebar controls
st.sidebar.header("⚙️ Pengaturan")
K = st.sidebar.number_input("Jumlah group (R01–Rxx)", min_value=2, max_value=30, value=12, step=1)
HARD_CAP = st.sidebar.number_input("Max toko per group (cap)", min_value=5, max_value=50, value=25, step=1)

st.sidebar.divider()
show_preview = st.sidebar.checkbox("Tampilkan preview tabel", value=True)
preview_rows = st.sidebar.slider("Jumlah baris preview", 5, 200, 30)

st.sidebar.divider()
st.sidebar.caption("Tip: Kalau hasil group terasa “kegedean”, kecilkan cap atau tambah jumlah group.")


# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Excel (.xlsx) dengan kolom: nama_toko | lat | long",
    type=["xlsx"],
)

# TAMPILKAN DUMMY + TEMPLATE SEBELUM UPLOAD
if not uploaded_file:
    render_template_section()
    st.info("Silakan upload file Excel untuk mulai proses grouping.")
    st.stop()


# =========================
# READ EXCEL
# =========================
try:
    df_raw = pd.read_excel(uploaded_file, sheet_name=0)
except Exception as e:
    st.error(f"Gagal baca Excel. Pastikan file .xlsx valid.\n\nDetail: {e}")
    st.stop()

# Preview
st.subheader("📄 Preview Data")
if show_preview:
    st.dataframe(df_raw.head(preview_rows), use_container_width=True)
else:
    st.caption("Preview dimatikan (aktifkan di sidebar jika perlu).")

# Validate
ok, msg, df_clean = validate_input_df(df_raw)
if not ok:
    st.error(msg)
    st.stop()
st.success("Format file OK. Siap diproses.")


# =========================
# PROCESS
# =========================
with st.spinner("Processing grouping & membuat peta..."):
    try:
        df_result, folium_map = process_excel(df_clean, hard_cap=int(HARD_CAP), K=int(K))
    except Exception as e:
        st.error(
            "Proses gagal. Biasanya karena kombinasi K & cap tidak memungkinkan "
            "atau data terlalu ekstrem.\n\n"
            f"Detail: {e}"
        )
        st.stop()


# =========================
# OUTPUTS
# =========================
st.subheader("✅ Ringkasan")
st.write(f"Total titik diproses: **{len(df_result):,}**")

summary = (
    df_result["kategori"]
    .value_counts()
    .sort_index()
    .rename_axis("kategori")
    .to_frame("jumlah")
)
st.dataframe(summary, use_container_width=True)

st.subheader("🗺️ Peta Grouping")
components.html(folium_map._repr_html_(), height=720, scrolling=True)

st.subheader("⬇️ Download")
st.download_button(
    label="Download Excel hasil grouping",
    data=df_to_excel_bytes(df_result, sheet_name="grouped"),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("Lihat tabel hasil (sample 200 baris)"):
    st.dataframe(df_result.head(200), use_container_width=True)

# Optional debug info
if SHOW_DEBUG:
    import grouping, inspect

    st.subheader("🧪 Debug")
    st.write("grouping.py loaded from:", grouping.__file__)
    st.write("process_excel from:", inspect.getsourcefile(grouping.process_excel))
