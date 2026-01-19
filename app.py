# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import streamlit.components.v1 as components

from grouping import validate_input_df, process_excel


# -----------------------------
# Helpers
# -----------------------------
REQUIRED_COLS = ["nama_toko", "lat", "long"]


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and coerce lat/long (support comma decimal)."""
    df = df.copy()

    # normalize headers
    df.columns = [str(c).strip().lower() for c in df.columns]

    # common alias mapping
    alias = {
        "nama toko": "nama_toko",
        "nama_toko": "nama_toko",
        "toko": "nama_toko",
        "outlet": "nama_toko",
        "nama_outlet": "nama_toko",
        "store": "nama_toko",

        "latitude": "lat",
        "lat": "lat",

        "longitude": "long",
        "lon": "long",
        "long": "long",
        "lng": "long",
    }
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    # trim text col
    if "nama_toko" in df.columns:
        df["nama_toko"] = df["nama_toko"].astype(str).str.strip()

    # handle decimal comma for lat/long (e.g., "-6,2031" -> "-6.2031")
    for c in ["lat", "long"]:
        if c in df.columns:
            s = df[c]
            if s.dtype == "object":
                s = (
                    s.astype(str)
                    .str.strip()
                    .str.replace(" ", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
            df[c] = pd.to_numeric(s, errors="coerce")

    return df


def render_template_section():
    st.markdown("### 📌 Format Excel yang wajib")
    st.write("File Excel (.xlsx) **harus punya 3 kolom** berikut (header bebas huruf besar/kecil):")
    st.code("nama_toko | lat | long", language="text")

    st.markdown("#### Contoh isi (dummy)")
    df_dummy = pd.DataFrame(
        {
            "nama_toko": ["Toko Sumber Rejeki", "Warung Bu Nita", "Kios Asep Jaya"],
            "lat": [-6.306400, -6.295120, -6.315880],
            "long": [107.149558, 107.162330, 107.141220],
        }
    )
    st.dataframe(df_dummy, use_container_width=True)

    st.info(
        "📍 **Pemisah desimal:** disarankan pakai **titik (.)**. "
        "Kalau file kamu pakai **koma (,)** seperti `-6,3064`, app ini akan otomatis mengubah ke format titik."
    )

    st.download_button(
        label="⬇️ Download template Excel",
        data=df_to_excel_bytes(df_dummy, sheet_name="template"),
        file_name="template_jks_store_grouping.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="JKS Store Grouping", layout="wide")
st.title("📍 JKS Store Grouping")

# Sidebar controls
st.sidebar.header("⚙️ Pengaturan")
K = st.sidebar.number_input("Jumlah group (R01–Rxx)", min_value=2, max_value=30, value=12, step=1)
HARD_CAP = st.sidebar.number_input("Max toko per group (cap)", min_value=5, max_value=50, value=25, step=1)

st.sidebar.divider()
show_preview = st.sidebar.checkbox("Tampilkan preview tabel", value=True)
preview_rows = st.sidebar.slider("Jumlah baris preview", 5, 200, 30)

st.sidebar.divider()
debug_mode = st.sidebar.checkbox("Debug mode", value=False)

uploaded_file = st.file_uploader(
    "Upload Excel (.xlsx) format kolom: nama_toko | lat | long",
    type=["xlsx"]
)

# If no upload: show instructions + template + stop
if not uploaded_file:
    render_template_section()
    st.info("Upload file Excel dulu untuk mulai proses grouping & peta.")
    st.stop()


# -----------------------------
# Read excel
# -----------------------------
try:
    df_raw = pd.read_excel(uploaded_file, sheet_name=0)
except Exception as e:
    st.error(f"Gagal baca Excel. Pastikan file .xlsx valid.\n\nDetail: {e}")
    st.stop()

# Normalize / pre-clean (handles comma decimal)
df_raw_norm = normalize_columns(df_raw)

if debug_mode:
    st.caption("Debug: kolom setelah normalisasi")
    st.write(list(df_raw_norm.columns))


# -----------------------------
# Preview
# -----------------------------
st.subheader("📄 Preview Data")
if show_preview:
    st.dataframe(df_raw_norm.head(preview_rows), use_container_width=True)
else:
    st.caption("Preview dimatikan (aktifkan di sidebar jika perlu).")


# -----------------------------
# Validate
# -----------------------------
ok, msg, df_clean = validate_input_df(df_raw_norm)
if not ok:
    # kasih bantuan tambahan biar user ngerti
    st.error(msg)
    st.markdown("#### Checklist cepat")
    st.write("- Pastikan ada kolom: `nama_toko`, `lat`, `long`")
    st.write("- `lat/long` harus angka (boleh koma, nanti diubah otomatis)")
    st.write("- Tidak boleh kosong semua di lat/long")
    st.stop()

st.success("Format file OK. Siap diproses.")


# -----------------------------
# Process
# -----------------------------
with st.spinner("Processing grouping & membuat peta..."):
    try:
        df_result, folium_map = process_excel(df_clean, hard_cap=int(HARD_CAP), K=int(K))
    except Exception as e:
        st.error(
            "Proses gagal. Biasanya karena kombinasi K & cap tidak memungkinkan atau data terlalu ekstrem.\n\n"
            f"Detail: {e}"
        )
        st.stop()


# -----------------------------
# Summary
# -----------------------------
st.subheader("✅ Ringkasan")
st.write(f"Total titik diproses: **{len(df_result):,}**")
st.dataframe(
    df_result["kategori"].value_counts().sort_index().rename_axis("kategori").to_frame("jumlah"),
    use_container_width=True
)

# -----------------------------
# Map
# -----------------------------
st.subheader("🗺️ Peta Grouping")
components.html(folium_map._repr_html_(), height=720, scrolling=True)

# -----------------------------
# Download
# -----------------------------
st.subheader("⬇️ Download")
st.download_button(
    label="Download Excel hasil grouping",
    data=df_to_excel_bytes(df_result, sheet_name="grouped"),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

with st.expander("Lihat tabel hasil (sample 200 baris)"):
    st.dataframe(df_result.head(200), use_container_width=True)
