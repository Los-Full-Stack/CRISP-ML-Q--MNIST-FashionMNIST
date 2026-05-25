import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="CNN Inference Portal", layout="centered", page_icon="🧠")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .main > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #ffd86f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }

    .hero-sub {
        text-align: center;
        color: rgba(255,255,255,0.6);
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }

    /* ── Model selector cards ── */
    div[data-testid="stVerticalBlock"]:has(> div > div > label[data-testid="stRadioLabel"]) {
        overflow: visible;
    }
    div[data-testid="stRadio"] > div {
        display: flex;
        gap: 0;
        flex-direction: column;
    }
    div[data-testid="stRadio"] > div > label {
        display: flex;
        padding: 0;
        margin: 0;
        background: transparent;
        border: none;
        border-radius: 0;
    }
    div[data-testid="stRadio"] > div > label > div:first-child {
        display: none;
    }

    .model-radio-group {
        display: grid !important;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }

    .model-card {
        background: rgba(255, 255, 255, 0.04);
        border: 2px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .model-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 18px;
        opacity: 0;
        transition: opacity 0.25s ease;
        background: linear-gradient(135deg, rgba(240,147,251,0.08) 0%, rgba(245,87,108,0.08) 100%);
    }
    .model-card:hover {
        border-color: rgba(255,255,255,0.15);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }
    .model-card:hover::before { opacity: 1; }

    .model-card.selected {
        border-color: #f5576c;
        background: rgba(245, 87, 108, 0.08);
        box-shadow: 0 0 30px rgba(245, 87, 108, 0.15);
    }
    .model-card.selected::before { opacity: 1; }

    .model-card .icon {
        font-size: 2.8rem;
        margin-bottom: 0.6rem;
        position: relative;
    }
    .model-card .name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.3rem;
        position: relative;
    }
    .model-card .desc {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.4);
        line-height: 1.4;
        position: relative;
    }
    .model-card .check {
        position: absolute;
        top: 12px;
        right: 14px;
        font-size: 1rem;
        opacity: 0;
        transition: opacity 0.2s ease;
    }
    .model-card.selected .check { opacity: 1; }

    /* ── hide the actual radio ── */
    .stRadio [role="radiogroup"] { visibility: hidden; height: 0; overflow: hidden; margin: 0; padding: 0; }

    .result-box {
        background: linear-gradient(135deg, rgba(240,147,251,0.12) 0%, rgba(245,87,108,0.12) 100%);
        border: 1px solid rgba(240,147,251,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .stButton button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 20px rgba(245, 87, 108, 0.3);
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(245, 87, 108, 0.5);
    }

    .stFileUploader section {
        border: 2px dashed rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 2rem;
        background: rgba(255,255,255,0.03);
    }
    .stFileUploader section:hover {
        border-color: rgba(240,147,251,0.4);
        background: rgba(240,147,251,0.05);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    div[data-testid="stMetric"] > div:first-child {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: rgba(255,255,255,0.5);
    }
    div[data-testid="stMetric"] > div:nth-child(2) {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255,255,255,0.06);
        border-radius: 100px;
        padding: 0.3rem 0.9rem;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
        font-weight: 500;
    }

    hr {
        border-color: rgba(255,255,255,0.06);
        margin: 1.5rem 0;
    }

    .stSpinner > div { color: #f5576c; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🧠 CNN Inference Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Elegí un modelo, subí una imagen y deja que la red neuronal la clasifique</div>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def cargar_modelo(nombre_archivo):
    try:
        return tf.keras.models.load_model(nombre_archivo)
    except Exception as e:
        st.error(f"No se pudo cargar '{nombre_archivo}'. Verificá que esté en la misma carpeta.")
        return None

# ── Model selection as visual cards ──

st.markdown("### 🎯 Seleccioná un modelo")

col_a, col_b = st.columns(2)

with col_a:
    mnist_selected = st.session_state.get("modelo_sel", "mnist") == "mnist"
    st.markdown(
        f'<div class="model-card {"selected" if mnist_selected else ""}" id="card-mnist">'
        f'<span class="check">{"✓" if mnist_selected else ""}</span>'
        f'<div class="icon">🔢</div>'
        f'<div class="name">MNIST Dígitos</div>'
        f'<div class="desc">Clasifica números escritos a mano del 0 al 9</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    if st.button("Usar MNIST", key="btn_mnist", use_container_width=True):
        st.session_state["modelo_sel"] = "mnist"
        st.rerun()

with col_b:
    fashion_selected = st.session_state.get("modelo_sel", "mnist") == "fashion"
    st.markdown(
        f'<div class="model-card {"selected" if fashion_selected else ""}" id="card-fashion">'
        f'<span class="check">{"✓" if fashion_selected else ""}</span>'
        f'<div class="icon">👕</div>'
        f'<div class="name">Fashion MNIST</div>'
        f'<div class="desc">Clasifica prendas de vestir y accesorios</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    if st.button("Usar Fashion MNIST", key="btn_fashion", use_container_width=True):
        st.session_state["modelo_sel"] = "fashion"
        st.rerun()

# ── Load the selected model ──

if st.session_state.get("modelo_sel", "mnist") == "mnist":
    modelo = cargar_modelo('modelo_mnist.keras')
    clases = [str(i) for i in range(10)]
else:
    modelo = cargar_modelo('modelo_fashion.keras')
    clases = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
              'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

st.markdown("---")

# ── Image upload ──

modelo_nombre = "MNIST Dígitos" if st.session_state.get("modelo_sel", "mnist") == "mnist" else "Fashion MNIST"
uploaded_file = st.file_uploader(
    f"📤 Subí una imagen para clasificar con **{modelo_nombre}**",
    type=["png", "jpg", "jpeg"],
    key=st.session_state.get("modelo_sel", "mnist")
)

if uploaded_file is not None and modelo is not None:
    imagen_original = Image.open(uploaded_file)

    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(imagen_original, caption="Imagen subida", width=220)

    with col_info:
        st.markdown(
            f'<div style="margin-bottom:0.5rem;color:rgba(255,255,255,0.5);font-size:0.85rem;">Pipeline de preprocesamiento</div>',
            unsafe_allow_html=True
        )

        imagen_gris = ImageOps.grayscale(imagen_original)
        imagen_reseteada = imagen_gris.resize((28, 28))
        matriz_pixeles = np.array(imagen_reseteada)

        borde_promedio = (matriz_pixeles[0,:].mean() + matriz_pixeles[-1,:].mean() +
                          matriz_pixeles[:,0].mean() + matriz_pixeles[:,-1].mean()) / 4
        if borde_promedio > 127:
            matriz_pixeles = 255 - matriz_pixeles

        tensor_entrada = matriz_pixeles.astype('float32') / 255.0
        tensor_entrada = np.expand_dims(tensor_entrada, axis=(0, -1))

        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.5rem;">'
            f'<span class="status-badge">📐 28×28 px</span>'
            f'<span class="status-badge">🌑 Escala de grises</span>'
            f'<span class="status-badge">{"🔁 Invertida" if borde_promedio > 127 else "✅ Normal"}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    if st.button("🚀 Ejecutar Predicción", key="btn_pred"):
        with st.spinner("La red neuronal está analizando los píxeles..."):
            predicciones = modelo.predict(tensor_entrada)
            id_clase = np.argmax(predicciones[0])
            probabilidad = predicciones[0][id_clase] * 100

        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Categoría Detectada", value=clases[id_clase])
        with col2:
            st.metric(label="Confianza del Modelo", value=f"{probabilidad:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📊 Distribución de Probabilidades")
        dict_prob = dict(zip(clases, [float(p) for p in predicciones[0]]))
        st.bar_chart(dict_prob, height=280)

        if probabilidad < 60:
            st.warning("⚠️ La confianza es baja. Probá con otra imagen más nítida.")
        elif probabilidad >= 90:
            st.success("✅ Predicción con alta confianza.")
else:
    st.markdown(
        '<div style="text-align:center;padding:3rem 1rem;color:rgba(255,255,255,0.3);">'
        '📤 Subí una imagen para comenzar</div>',
        unsafe_allow_html=True
    )
