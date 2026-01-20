# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import streamlit.components.v1 as components

from grouping import validate_input_df, process_excel

st.set_page_config(page_title="JKS Store Grouping", layout="wide")
st.title("📍 JKS Store Grouping")

# =========================
# Sidebar controls
# =========================
st.sidebar.header("⚙️ Pengaturan")
K = st.sidebar.number_input("Jumlah group (R01–Rxx)", min_value=2, max_value=30, value=12, step=1)
HARD_CAP = st.sidebar.number_input("Max toko per group (cap)", min_value=5, max_value=50, value=25, step=1)

st.sidebar.divider()
st.sidebar.subheader("🧠 Refinement (anti lompat)")
REFINE_ITER = st.sidebar.slider("Jumlah iterasi refinement", 0, 20, 6)
NEIGHBOR_K = st.sidebar.slider("Jumlah tetangga (k-NN)", 3, 30, 10)

st.sidebar.divider()
show_preview = st.sidebar.checkbox("Tampilkan preview tabel", value=True)
preview_rows = st.sidebar.slider("Jumlah baris preview", 5, 200, 30)


# =========================
# Helpers
# =========================
def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "template") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()


# =========================
# Landing (before upload)
# =========================
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

dummy = pd.DataFrame(
    {
        "nama_toko": ["TOKO ALFA", "TOKO BETA", "TOKO GAMMA"],
        "lat": [-6.2001, -6.2015, -6.1989],
        "long": [106.8167, 106.8201, 106.8122],
    }
)

c1, c2 = st.columns([2, 1], vertical_alignment="top")
with c1:
    st.caption("Contoh data dummy:")
    st.dataframe(dummy, use_container_width=True)
with c2:
    st.caption("Template:")
    st.download_button(
        "⬇️ Download Template Excel",
        data=df_to_excel_bytes(dummy),
        file_name="template_jks_store_grouping.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.divider()

uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload file Excel dulu untuk mulai proses grouping.")
    st.stop()


# =========================
# Read excel
# =========================
try:
    df_raw = pd.read_excel(uploaded_file, sheet_name=0)
except Exception as e:
    st.error(f"Gagal baca Excel. Pastikan file .xlsx valid.\n\nDetail: {e}")
    st.stop()


# Preview
st.subheader("📄 Preview Data Upload")
if show_preview:
    st.dataframe(df_raw.head(preview_rows), use_container_width=True)
else:
    st.caption("Preview dimatikan (aktifkan di sidebar jika perlu).")

# Validate
ok, msg, df_clean = validate_input_df(df_raw)
if not ok:
    st.error(msg)
    st.stop()
st.success(msg)


# =========================
# Process
# =========================
with st.spinner("Processing grouping & membuat peta..."):
    try:
        df_result, folium_map = process_excel(
            df_clean,
            hard_cap=int(HARD_CAP),
            K=int(K),
            refine_iter=int(REFINE_ITER),
            neighbor_k=int(NEIGHBOR_K),
        )
    except Exception as e:
        st.error(
            "Proses gagal. Biasanya karena kombinasi K & cap tidak memungkinkan.\n\n"
            f"Detail: {e}"
        )
        st.stop()


# Summary
st.subheader("✅ Ringkasan")
st.write(f"Total titik diproses: **{len(df_result):,}**")
st.dataframe(
    df_result["kategori"].value_counts().sort_index().rename_axis("kategori").to_frame("jumlah"),
    use_container_width=True,
)

# Map
st.subheader("🗺️ Peta Grouping")
components.html(folium_map._repr_html_(), height=720, scrolling=True)

# Download
st.subheader("⬇️ Download")
st.download_button(
    "Download Excel hasil grouping",
    data=df_to_excel_bytes(df_result, sheet_name="grouped"),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("Lihat tabel hasil (sample 200 baris)"):
    st.dataframe(df_result.head(200), use_container_width=True)
