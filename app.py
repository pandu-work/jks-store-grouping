# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import streamlit.components.v1 as components

from grouping import validate_input_df, process_excel
import grouping, inspect
st.write("grouping.py loaded from:", grouping.__file__)
st.write("process_excel from:", inspect.getsourcefile(grouping.process_excel))

st.set_page_config(page_title="JKS Store Grouping", layout="wide")
st.title("📍 JKS Store Grouping")

# Sidebar controls
st.sidebar.header("⚙️ Pengaturan")
K = st.sidebar.number_input("Jumlah group (R01–Rxx)", min_value=2, max_value=30, value=12, step=1)
HARD_CAP = st.sidebar.number_input("Max toko per group (cap)", min_value=5, max_value=50, value=25, step=1)

st.sidebar.divider()
show_preview = st.sidebar.checkbox("Tampilkan preview tabel", value=True)
preview_rows = st.sidebar.slider("Jumlah baris preview", 5, 200, 30)

uploaded_file = st.file_uploader("Upload Excel (.xlsx) format kolom: nama_toko | lat | long", type=["xlsx"])

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="grouped")
    return bio.getvalue()

if not uploaded_file:
    st.info("Upload file Excel dulu untuk mulai.")
    st.stop()

# Read excel
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

# Process
with st.spinner("Processing grouping & membuat peta..."):
    try:
        df_result, folium_map = process_excel(df_clean, hard_cap=int(HARD_CAP), K=int(K))
    except Exception as e:
        st.error("Proses gagal. Biasanya karena kombinasi K & cap tidak memungkinkan atau data terlalu ekstrem.\n\n"
                 f"Detail: {e}")
        st.stop()

# Summary
st.subheader("✅ Ringkasan")
st.write(f"Total titik diproses: **{len(df_result):,}**")
st.dataframe(
    df_result["kategori"].value_counts().sort_index().rename_axis("kategori").to_frame("jumlah"),
    use_container_width=True
)

# Map
st.subheader("🗺️ Peta Grouping")
components.html(folium_map._repr_html_(), height=720, scrolling=True)

# Download
st.subheader("⬇️ Download")
st.download_button(
    label="Download Excel hasil grouping",
    data=df_to_excel_bytes(df_result),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

with st.expander("Lihat tabel hasil (sample 200 baris)"):
    st.dataframe(df_result.head(200), use_container_width=True)
