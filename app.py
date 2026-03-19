from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

# CONFIG
st.set_page_config(
    page_title="Pink AI Object Detection",
    page_icon="💖",
    layout="wide"
)

# 🎨 ESTILOS ROSADOS
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #ffe4ec, #fff1f5);
    color: #4b2e2e;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #b83280;
}
p, label, span {
    color: #6b4c4c;
}
section[data-testid="stSidebar"] {
    background: #fff0f6;
}
section[data-testid="stSidebar"] * {
    color: #b83280 !important;
}
.stButton>button {
    background: linear-gradient(90deg, #ff8fab, #ffb3c6);
    color: white;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# MODELO
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# HEADER
st.markdown("## 💖 Pink AI Object Detection")
st.caption("Detecta objetos con estilo ✨🌸")

# CARGAR MODELO
with st.spinner("Cargando modelo..."):
    model = load_model()

# SI EL MODELO NO CARGA → STOP LIMPIO
if model is None:
    st.error("❌ No se pudo cargar el modelo")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Configuración")
    conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25)
    iou_threshold = st.slider("Umbral IoU", 0.0, 1.0, 0.45)
    max_det = st.number_input("Máx. detecciones", 10, 2000, 300)

# INPUT
st.markdown("### 📸 Captura una imagen")
picture = st.camera_input("")

if picture is not None:

    bytes_data = picture.getvalue()
    pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    np_img = np.array(pil_img)[..., ::-1]

    with st.spinner("🔍 Analizando imagen..."):
        try:
            results = model(
                np_img,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=int(max_det)
            )
        except Exception as e:
            st.error(f"Error durante la detección: {str(e)}")
            st.stop()

    result = results[0]
    boxes = result.boxes
    annotated = result.plot()
    annotated_rgb = annotated[:, :, ::-1]

    col1, col2 = st.columns([1.5, 1])

    # IMAGEN
    with col1:
        st.markdown("### 🖼️ Resultado")
        st.image(annotated_rgb, use_container_width=True)

    # RESULTADOS
    with col2:
        st.markdown("### 📊 Análisis")

        if boxes is not None and len(boxes) > 0:

            label_names = model.names
            category_count = {}
            category_conf = {}

            for box in boxes:
                cat = int(box.cls.item())
                conf = float(box.conf.item())

                category_count[cat] = category_count.get(cat, 0) + 1
                category_conf.setdefault(cat, []).append(conf)

            total_objects = sum(category_count.values())
            st.metric("Objetos detectados", total_objects)

            data = []
            for cat, count in category_count.items():
                data.append({
                    "Categoría": label_names[cat],
                    "Cantidad": count,
                    "Confianza": round(np.mean(category_conf[cat]), 2)
                })

            df = pd.DataFrame(data)

            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index("Categoría")["Cantidad"])

        else:
            st.warning("😕 No se detectaron objetos")
            st.caption("Intenta bajar la confianza en la barra lateral")

# FOOTER
st.markdown("---")
st.caption("💖 Hecho con Streamlit + YOLOv5")

st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")
