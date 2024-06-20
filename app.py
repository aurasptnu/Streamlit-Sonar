import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Memuat dataset
sonar_data = pd.read_csv('sonar data.csv', header=None)

# Judul aplikasi Streamlit
st.title('Identifikasi Objek Sonar')

# Menampilkan dataset
if st.checkbox('Tampilkan data mentah'):
    st.write(sonar_data)

# Pemrosesan data
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Melatih model Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Akurasi pada data pelatihan dan pengujian
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

st.write(f'Akurasi pada data pelatihan: {training_data_accuracy}')
st.write(f'Akurasi pada data pengujian: {test_data_accuracy}')

# Input data untuk prediksi
st.header('Prediksi apakah objek adalah Batu atau Ranjau')
input_data = []
for i in range(60):
    value = st.number_input(f'Masukkan nilai untuk fitur {i+1}', min_value=0.0, max_value=1.0, step=0.001, value=0.0)
    input_data.append(value)

if st.button('Prediksi'):
    # Mengubah input_data menjadi array numpy
    input_data_as_numpy_array = np.asarray(input_data)

    # Merubah bentuk array untuk prediksi satu instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)
    if prediction[0] == 'R':
        st.write('Objek tersebut adalah Batu')
        st.write('Objek tersebut adalah Ranjau')
