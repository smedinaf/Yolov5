from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

# CONFIG
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯",
    layout="wide"
)

# CSS PERSONALIZADO
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
h1, h2, h3 {
    color: #f8fafc;
}
.stButton>button {
    background: #22c55e;
    color: white;
    border-radius: 10px;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 10px;
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
st.markdown("## 🎯 AI Object Detection")
st.caption("Detecta objetos en tiempo real usando YOLOv5 🚀")

# LOAD MODEL
with st.spinner("Cargando modelo..."):
    model = load_model()

if model:

    # SIDEBAR
    with st.sidebar:
        st.title("⚙️ Configuración")
        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25)
        iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45)
        max_det        = st.number_input("Máx. detecciones", 10, 2000, 300)

    # INPUT
    st.markdown("### 📸 Captura una imagen")
    picture = st.camera_input("")

    if picture:

        bytes_data = picture.getvalue()
        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1]

        with st.spinner("🔍 Analizando imagen..."):
            results = model(
                np_img,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=int(max_det)
            )

        result    = results[0]
        boxes     = result.boxes
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
                label_names    = model.names
                category_count = {}
                category_conf  = {}

                for box in boxes:
                    cat  = int(box.cls.item())
                    conf = float(box.conf.item())
                    category_count[cat] = category_count.get(cat, 0) + 1
                    category_conf.setdefault(cat, []).append(conf)

                total_objects = sum(category_count.values())

                # MÉTRICAS
                st.metric("Objetos detectados", total_objects)

                # DATA
                data = [
                    {
                        "Categoría": label_names[cat],
                        "Cantidad": count,
                        "Confianza": round(np.mean(category_conf[cat]), 2)
                    }
                    for cat, count in category_count.items()
                ]

                df = pd.DataFrame(data)

                st.dataframe(df, use_container_width=True)

                st.markdown("### 📈 Distribución")
                st.bar_chart(df.set_index("Categoría")["Cantidad"])

            else:
                st.warning("😕 No se detectaron objetos")
                st.caption("Intenta bajar la confianza en la barra lateral")

else:
    st.error("❌ No se pudo cargar el modelo")
    st.stop()

st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")
