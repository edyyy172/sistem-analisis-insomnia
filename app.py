import streamlit as st
import pandas as pd
import joblib
import os

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================

st.set_page_config(
    page_title="Sistem Identifikasi Gangguan Tidur",
    page_icon="",
    layout="centered"
)

st.title(" Sistem Identifikasi Gangguan Tidur")

st.write("""
Aplikasi ini menggunakan algoritma **K-Nearest Neighbors (KNN)**
untuk mengidentifikasi kemungkinan gangguan tidur berdasarkan
karakteristik pengguna.
""")

# ======================================================
# LOAD MODEL
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("Model tidak berhasil dimuat.")
    st.exception(e)
    st.stop()

# ======================================================
# INPUT
# ======================================================

st.header("Masukkan Data")

age = st.number_input(
    "Umur (Tahun)",
    min_value=18,
    max_value=80,
    value=25
)

sleep_duration = st.number_input(
    "Durasi Tidur (Jam)",
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

quality = st.slider(
    "Kualitas Tidur",
    min_value=1,
    max_value=10,
    value=6
)

physical = st.number_input(
    "Aktivitas Fisik (Menit/Hari)",
    min_value=0,
    max_value=120,
    value=45
)

stress = st.slider(
    "Tingkat Stres",
    min_value=1,
    max_value=10,
    value=5
)

# ======================================================
# VALIDASI
# ======================================================

if sleep_duration < 5:
    st.warning(
        "⚠ Durasi tidur yang Anda masukkan berada di luar rentang data pelatihan model. "
        "Hasil prediksi mungkin kurang akurat."
    )

# ======================================================
# PREDIKSI
# ======================================================

if st.button("🔍 Prediksi"):

    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Sleep Duration": [sleep_duration],
            "Quality of Sleep": [quality],
            "Physical Activity Level": [physical],
            "Stress Level": [stress],
        }
    )

    hasil = model.predict(input_data)[0]

    label = {
        0: "Normal",
        1: "Insomnia",
        2: "Sleep Apnea"
    }

    hasil_prediksi = label.get(hasil, "Tidak Diketahui")

    st.divider()

    st.subheader("Hasil Prediksi")

    if hasil == 0:

        st.success("✅ Kondisi : NORMAL")

        st.markdown(f"""
### Interpretasi

Model KNN memprediksi bahwa kondisi tidur Anda termasuk **{hasil_prediksi}**.

### Saran

- Pertahankan durasi tidur sekitar **7–9 jam**.
- Pertahankan kualitas tidur yang baik.
- Tetap aktif berolahraga.
- Kelola stres dengan baik.
- Lakukan pemeriksaan kesehatan secara berkala.
""")

    elif hasil == 1:

        st.warning("⚠ Kondisi : INSOMNIA")

        st.markdown(f"""
### Interpretasi

Model KNN memprediksi bahwa Anda memiliki kecenderungan mengalami **{hasil_prediksi}**.

### Saran

- Tidur dan bangun pada waktu yang sama setiap hari.
- Hindari kafein pada malam hari.
- Kurangi penggunaan gadget sebelum tidur.
- Lakukan relaksasi sebelum tidur.
- Konsultasikan dengan tenaga kesehatan apabila keluhan berlangsung lama.
""")

    else:

        st.error("🚨 Kondisi : SLEEP APNEA")

        st.markdown(f"""
### Interpretasi

Model KNN memprediksi bahwa Anda memiliki kecenderungan mengalami **{hasil_prediksi}**.

### Saran

- Segera berkonsultasi dengan dokter.
- Hindari merokok.
- Jaga berat badan ideal.
- Lakukan pemeriksaan tidur apabila diperlukan.
""")

    # ======================================================
    # RINGKASAN INPUT
    # ======================================================

    st.divider()

    st.subheader("Data yang Dimasukkan")

    hasil_input = pd.DataFrame({
        "Umur": [age],
        "Durasi Tidur (Jam)": [sleep_duration],
        "Kualitas Tidur": [quality],
        "Aktivitas Fisik": [physical],
        "Tingkat Stres": [stress]
    })

    st.dataframe(hasil_input, use_container_width=True)

st.divider()

st.caption("Metode : K-Nearest Neighbors (KNN)")