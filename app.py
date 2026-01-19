# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import streamlit.components.v1 as components
import inspect

from grouping import validate_input_df, process_excel


# =========================
# Helpers
# =========================
def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "sheet1") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()


def make_template_df() -> pd.DataFrame:
    # Dummy example: sengaja pakai titik (.) karena paling aman
    return pd.DataFrame(
        {
            "nama_toko": ["TOKO MAJU JAYA", "TOKO BAROKAH", "TOKO SUMBER REZEKI", "TOKO NUSANTARA"],
            "lat": [-6.917464, -6.914500, -6.920100, -6.910900],
            "long": [107.619123, 107.610200, 107.623800, 107.635400],
        }
    )


def normalize_latlong_like_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bantu kasus Excel yang kebaca sebagai string dan pakai koma:
    -6,917464 -> -6.917464
    """
    df2 = df.copy()
    for col in ["lat", "long"]:
        if col in df2.columns:
            # ubah ke string dulu, ganti koma jadi titik, lalu to_numeric
            df2[col] = (
                df2[col]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
            )
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
    return df2


# =========================
# UI
# =========================
st.set_page_config(page_title="JKS Store Grouping", layout="wide")
st.title("📍 JKS Store Grouping")

# Sidebar controls
st.sidebar.header("⚙️ Pengaturan")
K = st.sidebar.number_input("Jumlah group (R01–Rxx)", min_value=2, max_value=30, value=12, step=1)
HARD_CAP = st.sidebar.number_input("Max toko per group (cap)", min_value=5, max_value=50, value=25, step=1)

st.sidebar.divider()
show_preview = st.sidebar.checkbox("Tampilkan preview tabel", value=True)
preview_rows = st.sidebar.slider("Jumlah baris preview", 5, 200, 30)

# Optional refinement controls (kalau process_excel support)
st.sidebar.divider()
st.sidebar.subheader("🧠 Refinement (anti “lompat”)")
use_refine = st.sidebar.checkbox("Aktifkan refinement", value=True)
refine_iter = st.sidebar.slider("Jumlah iterasi refinement", 0, 20, 6, help="0 = mati (lebih cepat), makin tinggi makin rapi tapi lebih lama")
neighbor_k = st.sidebar.slider("Tetangga terdekat (k)", 3, 20, 7, help="Dipakai untuk merapikan group berdasarkan kedekatan lokal")

# Pre-upload 안내 / instructions
st.subheader("📌 Format Excel yang dibutuhkan (sebelum upload)")

st.markdown(
    """
**Kolom wajib (case-insensitive):**
- `nama_toko`  → teks bebas (nama outlet)
- `lat`        → latitude (angka)
- `long`       → longitude (angka)

**Contoh format nilai:**
- Disarankan pakai **titik** untuk desimal: `-6.917464`
- Kalau file kamu pakai **koma**: `-6,917464` → aman, app ini akan coba auto-konversi.

**Catatan penting:**
- Sheet yang dipakai adalah **sheet pertama** (sheet index 0).
- Kalau ada baris lat/long kosong / non-angka, sistem akan menolak dan kasih error.
"""
)

template_df = make_template_df()
st.dataframe(template_df, use_container_width=True)

st.download_button(
    label="⬇️ Download Template Excel (contoh)",
    data=df_to_excel_bytes(template_df, sheet_name="template"),
    file_name="template_jks_store_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Excel (.xlsx) — kolom wajib: nama_toko | lat | long",
    type=["xlsx"]
)

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

# Normalize headers (case-insensitive)
df_raw.columns = [str(c).strip() for c in df_raw.columns]

# Try normalize lat/long if present (string commas -> dots)
df_raw = normalize_latlong_like_strings(df_raw)

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
# Process
# =========================
with st.spinner("Processing grouping & membuat peta..."):
    try:
        # Panggil process_excel dengan argumen yang tersedia saja (biar backward-compatible)
        sig = inspect.signature(process_excel)
        kwargs = {"hard_cap": int(HARD_CAP), "K": int(K)}

        if use_refine and refine_iter > 0:
            if "refine_iter" in sig.parameters:
                kwargs["refine_iter"] = int(refine_iter)
            if "neighbor_k" in sig.parameters:
                kwargs["neighbor_k"] = int(neighbor_k)

        df_result, folium_map = process_excel(df_clean, **kwargs)

    except Exception as e:
        st.error(
            "Proses gagal. Biasanya karena kombinasi K & cap tidak memungkinkan, atau ada data ekstrem.\n\n"
            f"Detail: {e}"
        )
        st.stop()

# =========================
# Summary
# =========================
st.subheader("✅ Ringkasan")
st.write(f"Total titik diproses: **{len(df_result):,}**")

if "kategori" in df_result.columns:
    st.dataframe(
        df_result["kategori"].value_counts().sort_index().rename_axis("kategori").to_frame("jumlah"),
        use_container_width=True
    )
else:
    st.warning("Kolom 'kategori' tidak ditemukan di hasil. Cek output process_excel().")

# =========================
# Map
# =========================
st.subheader("🗺️ Peta Grouping")
components.html(folium_map._repr_html_(), height=720, scrolling=True)

# =========================
# Download
# =========================
st.subheader("⬇️ Download")
st.download_button(
    label="Download Excel hasil grouping",
    data=df_to_excel_bytes(df_result, sheet_name="grouped"),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("Lihat tabel hasil (sample 200 baris)"):
    st.dataframe(df_result.head(200), use_container_width=True)
