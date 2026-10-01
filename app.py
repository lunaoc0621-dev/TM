import os
import platform

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from keras.models import load_model

st.write("Versión de Python:", platform.python_version())

# ---------------------------------------------------------------
# Modelo y etiquetas
# ---------------------------------------------------------------
# El modelo (keras_model.h5) tiene 2 salidas (softmax de 2 neuronas).
# El .h5 no guarda los nombres de las clases; Teachable Machine los
# entrega aparte en labels.txt. Si ese archivo existe se usa; si no,
# se usan las etiquetas que ya tenías en tu código.
DEFAULT_LABELS = ["Izquierda", "Arriba"]


@st.cache_resource
def cargar_modelo():
    return load_model("keras_model.h5", compile=False)


def cargar_etiquetas(n_clases):
    if os.path.exists("labels.txt"):
        with open("labels.txt", encoding="utf-8") as f:
            etiquetas = []
            for linea in f.read().splitlines():
                if not linea.strip():
                    continue
                partes = linea.strip().split(" ", 1)
                # Formato de Teachable Machine: "0 Nombre"
                etiquetas.append(partes[1] if len(partes) == 2 and partes[0].isdigit() else linea.strip())
        if len(etiquetas) == n_clases:
            return etiquetas
    if len(DEFAULT_LABELS) == n_clases:
        return DEFAULT_LABELS
    return [f"Clase {i}" for i in range(n_clases)]


model = cargar_modelo()
n_clases = model.output_shape[-1]
labels = cargar_etiquetas(n_clases)

# ---------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------
st.title("Reconocimiento de Imágenes")

if os.path.exists("OIG5.jpg"):
    st.image(Image.open("Luna.jpg"), width=350)

with st.sidebar:
    st.subheader(
        "Usando un modelo entrenado en Teachable Machine puedes usarlo "
        "en esta app para identificar"
    )
    st.write("Clases detectables:")
    for i, nombre in enumerate(labels):
        st.write(f"{i}. {nombre}")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer).convert("RGB")

    # Mismo preprocesamiento que Teachable Machine: recorte central 224x224
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img)

    # Normalizar a [-1, 1]
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Inferencia
    prediction = model.predict(data)[0]

    # Clase ganadora
    idx = int(np.argmax(prediction))
    st.header(f"{labels[idx]}, con probabilidad: {prediction[idx]:.2%}")

    # Probabilidad de todas las clases
    st.subheader("Probabilidades por clase")
    for nombre, prob in zip(labels, prediction):
        st.write(f"{nombre}: {prob:.2%}")
        st.progress(float(prob))

