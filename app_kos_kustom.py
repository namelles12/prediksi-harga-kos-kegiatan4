import streamlit as st
import pandas as pd
import joblib

st.title("🏠 Sistem Prediksi Harga Kos Madiun")
st.write("NIM: 253307044")

@st.cache_resource
def muat_model():
    return joblib.load("model_kos.joblib")

model = muat_model()

url = "https://raw.githubusercontent.com/namelles12/sistem-cerdas-dataset/refs/heads/main/dataset/kos_madiun.csv"

@st.cache_data
def muat_data():
    return pd.read_csv(url)

df = muat_data()

jarak = st.slider("Jarak ke kampus (km)", 0.3, 5.0, 1.0, 0.1)
rating = st.slider("Rating kos", 3.0, 5.0, 4.0, 0.1)
ac = st.selectbox("Fasilitas AC", ["ya", "tidak"])
penghuni = st.slider("Jumlah penghuni kos", 1, 5, 1)

if st.button("🔮 Prediksi Harga"):
    data_baru = pd.DataFrame({
        "jarak_km": [jarak],
        "rating": [rating],
        "ac_num": [1 if ac == "ya" else 0]
    })
    
    st.subheader("📊 Rata-rata Harga Kos Berdasarkan Fasilitas AC")

    rata_rata = df.groupby("ac")["harga"].mean()

    st.bar_chart(rata_rata)

    harga = model.predict(data_baru)[0]

    st.success(f"Perkiraan harga: Rp {harga:.0f} ribu / bulan")
