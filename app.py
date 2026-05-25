import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import altair as alt
import io, base64

st.set_page_config(page_title="CNN Inference Portal", layout="wide", page_icon="🧠")

# ── Palette ──
BG = "#070b18"
SURFACE = "#0e1428"
SURFACE2 = "#161d3a"
BORDER = "#1c2648"
PRI = "#38bdf8"
SEC = "#818cf8"
SUCCESS = "#22c55e"
WARN = "#eab308"
DANGER = "#ef4444"
TEXT = "#e2e8f0"
MUTED = "#64748b"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
    .stApp {{ background: {BG}; }}
    .block-container {{ max-width: 90% !important; padding-top: 0.8rem !important; padding-bottom: 0.8rem !important; }}
    .main > div {{ background: transparent; border: none; padding: 0; margin: 0; }}
    section[data-testid="stSidebar"] {{ display: none; }}

    .stButton button {{
        background: linear-gradient(135deg, {PRI} 0%, {SEC} 100%);
        color: white; font-weight: 600; border: none; border-radius: 10px;
        padding: 0.45rem 1.6rem; font-size: 0.9rem; transition: all 0.2s ease;
        box-shadow: 0 4px 16px rgba(56,189,248,0.2); width: 100%;
    }}
    .stButton button:hover {{ transform: translateY(-1px); box-shadow: 0 6px 24px rgba(56,189,248,0.35); }}

    .stFileUploader section {{
        border: 2px dashed {BORDER}; border-radius: 12px; padding: 1rem 1.4rem;
        background: {SURFACE2}44; transition: all 0.2s;
    }}
    .stFileUploader section:hover {{ border-color: {PRI}66; background: rgba(56,189,248,0.04); }}
    .stFileUploader section ul {{ padding: 0; margin: 0.4rem 0 0 0; }}
    .stFileUploader section ul li {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
        padding: 0.35rem 0.7rem; margin-bottom: 0.25rem; list-style: none;
    }}
    .stFileUploader section ul li span {{ font-size: 0.85rem !important; color: {TEXT} !important; }}
    .stFileUploader section ul li small {{ font-size: 0.75rem !important; color: {MUTED} !important; }}

    .stSpinner > div {{ color: {PRI}; }}
    hr {{ border-color: {BORDER}; margin: 0.6rem 0; }}

    .card {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 1.1rem 1.5rem; margin-bottom: 0.8rem;
    }}
    .card-title {{
        font-size: 1rem; font-weight: 600; color: {TEXT};
        margin-bottom: 0.2rem; display: flex; align-items: center; gap: 0.5rem;
    }}
    .card-sub {{
        font-size: 0.75rem; color: {MUTED}; margin-bottom: 0.6rem;
    }}

    .sel-card {{
        background: linear-gradient(145deg, {SURFACE2} 0%, {SURFACE} 100%);
        border: 1px solid {BORDER}; border-radius: 18px;
        padding: 2.2rem 1.5rem; text-align: center; cursor: pointer;
        transition: all 0.3s ease; height: 100%;
    }}
    .sel-card:hover {{
        border-color: {PRI}; transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(56,189,248,0.08);
    }}
    .sel-card .ico {{ font-size: 3.8rem; margin-bottom: 0.6rem; }}
    .sel-card .nm {{ font-size: 1.3rem; font-weight: 700; color: {TEXT}; margin-bottom: 0.3rem; }}
    .sel-card .dc {{ font-size: 0.85rem; color: {MUTED}; line-height: 1.5; }}

    .kpi-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 0.6rem;
        margin-bottom: 0.6rem;
    }}
    .kpi {{
        text-align: center; padding: 0.7rem 0.2rem;
        background: {SURFACE2}55; border-radius: 10px; border: 1px solid {BORDER};
    }}
    .kpi .nm {{ font-size: 1.8rem; font-weight: 800; color: #fff; line-height: 1.1; }}
    .kpi .nm.blue {{ color: {PRI}; }}
    .kpi .nm.green {{ color: {SUCCESS}; }}
    .kpi .nm.yellow {{ color: {WARN}; }}
    .kpi .nm.red {{ color: {DANGER}; }}
    .kpi .lb {{ font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.08em;
               color: {MUTED}; margin-top: 0.1rem; font-weight: 500; }}

    .res-card {{
        background: {SURFACE2}77; border: 1px solid {BORDER};
        border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.5rem;
    }}
    .rl {{ font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED}; margin-bottom: 0.1rem; }}
    .rcat {{ display: inline-block; background: rgba(56,189,248,0.12); color: {PRI}; font-weight: 600; font-size: 0.85rem; padding: 0.1rem 0.7rem; border-radius: 6px; }}
    .rval {{ font-size: 1.3rem; font-weight: 700; color: {TEXT}; line-height: 1.2; }}
    .rval.green {{ color: {SUCCESS}; }} .rval.yellow {{ color: {WARN}; }} .rval.red {{ color: {DANGER}; }}
    .rbar {{ width:100%; height:5px; background:{BG}; border-radius:3px; overflow:hidden; margin:0.15rem 0 0.2rem 0; }}
    .rbar-fill {{ height:100%; border-radius:3px; }}
    .rbar-fill.green {{ background:{SUCCESS}; }} .rbar-fill.yellow {{ background:{WARN}; }} .rbar-fill.red {{ background:{DANGER}; }}
    .badge {{
        display: inline-flex; align-items: center; gap: 0.25rem;
        background: {BG}; border: 1px solid {BORDER};
        border-radius: 100px; padding: 0.1rem 0.5rem;
        font-size: 0.65rem; color: {TEXT}bb; font-weight: 500;
        cursor: help;
    }}
    .msg {{
        font-size: 0.85rem; padding: 0.35rem 0.8rem; border-radius: 8px; margin-top: 0.3rem;
    }}
    .msg.ok {{ background: rgba(34,197,94,0.08); color: {SUCCESS}; border: 1px solid rgba(34,197,94,0.15); }}
    .msg.info {{ background: rgba(56,189,248,0.08); color: {PRI}; border: 1px solid rgba(56,189,248,0.15); }}
    .msg.warn {{ background: rgba(234,179,8,0.08); color: {WARN}; border: 1px solid rgba(234,179,8,0.15); }}

    .hero {{
        text-align: center; padding: 1.2rem 0 0.3rem 0;
    }}
    .hero h1 {{
        font-size: 2.8rem; font-weight: 800; color: {TEXT};
        letter-spacing: -0.02em; margin: 0; line-height: 1.2;
        background: linear-gradient(135deg, {PRI}, {SEC});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .hero p {{ color: {MUTED}; font-size: 0.95rem; margin: 0.2rem 0 0 0; }}

    /* lightbox */
    .lb-overlay {{
        position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 999;
        display: flex; align-items: center; justify-content: center;
    }}
    .lb-overlay img {{ max-width: 90vw; max-height: 90vh; border-radius: 12px; }}
</style>
""", unsafe_allow_html=True)

# ── State ──
if "page" not in st.session_state:
    st.session_state.page = "select"
if "modelo_sel" not in st.session_state:
    st.session_state.modelo_sel = None
if "resultados" not in st.session_state:
    st.session_state.resultados = None

# ── Models ──
MODELOS = {
    "mnist": {
        "archivo": "modelo_mnist.keras",
        "nombre": "MNIST Dígitos",
        "icono": "🔢",
        "desc": "Reconoce números escritos a mano del 0 al 9",
        "detalle": "Ideal para dígitos numéricos escritos en papel, pizarras o formularios.",
        "clases": [str(i) for i in range(10)],
    },
    "fashion": {
        "archivo": "modelo_fashion.keras",
        "nombre": "Fashion MNIST",
        "icono": "👕",
        "desc": "Clasifica prendas de vestir y accesorios",
        "detalle": "Reconoce 10 categorías: camisetas, pantalones, zapatos, bolsos y más.",
        "clases": ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
                   'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín'],
    },
}

# ── Helpers ──

@st.cache_resource
def cargar_modelo(nombre_archivo):
    try:
        return tf.keras.models.load_model(nombre_archivo)
    except:
        return None

def preprocess_image(imagen_original):
    gris = ImageOps.grayscale(imagen_original)
    redim = gris.resize((28, 28))
    pix = np.array(redim)
    borde = (pix[0,:].mean() + pix[-1,:].mean() + pix[:,0].mean() + pix[:,-1].mean()) / 4
    invertida = borde > 127
    if invertida:
        pix = 255 - pix
    tensor = pix.astype('float32') / 255.0
    tensor = np.expand_dims(tensor, axis=(0, -1))
    return tensor, invertida, redim

def img_to_b64(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def conf_cls(pct):
    if pct >= 80: return "green"
    if pct >= 50: return "yellow"
    return "red"

def make_csv(resultados, clases):
    rows = []
    for r in resultados:
        probs_list = [float(p) * 100 for p in r["predicciones"]]
        row = {
            "Imagen": r["nombre"],
            "Categoría": r["clase"],
            "ID Clase": r["id_clase"],
            "Confianza (%)": round(r["probabilidad"], 2),
            "Invertida": "Sí" if r["invertida"] else "No",
        }
        for i, c in enumerate(clases):
            row[f"Prob {c} (%)"] = round(probs_list[i], 2)
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

# ══════════════════════════════════════════════════
#  PAGE: SELECT MODEL
# ══════════════════════════════════════════════════

if st.session_state.page == "select":
    st.markdown("""
    <div class="hero">
        <h1>🧠 CNN Inference Portal</h1>
        <p>Clasificación inteligente de imágenes con redes neuronales</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="card" style="padding:1.5rem 2rem;">'
        f'<div class="card-title" style="font-size:1.15rem;justify-content:center;">🎯 Elegí un modelo de clasificación</div>'
        f'<div class="card-sub" style="text-align:center;">Cada modelo fue entrenado para reconocer un tipo distinto de imagen. Seleccioná el que mejor se adapte a lo que querés clasificar.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        info = MODELOS["mnist"]
        st.markdown(
            f'<div class="sel-card">'
            f'<div class="ico">{info["icono"]}</div>'
            f'<div class="nm">{info["nombre"]}</div>'
            f'<div class="dc">{info["desc"]}</div>'
            f'<div style="margin-top:0.6rem;font-size:0.75rem;color:{MUTED};">{info["detalle"]}</div>'
            f'</div>', unsafe_allow_html=True
        )
        if st.button(f"Usar {info['nombre']}", key="sel_mnist", use_container_width=True):
            st.session_state.modelo_sel = "mnist"
            st.session_state.resultados = None
            st.session_state.page = "classify"
            st.rerun()

    with c2:
        info = MODELOS["fashion"]
        st.markdown(
            f'<div class="sel-card">'
            f'<div class="ico">{info["icono"]}</div>'
            f'<div class="nm">{info["nombre"]}</div>'
            f'<div class="dc">{info["desc"]}</div>'
            f'<div style="margin-top:0.6rem;font-size:0.75rem;color:{MUTED};">{info["detalle"]}</div>'
            f'</div>', unsafe_allow_html=True
        )
        if st.button(f"Usar {info['nombre']}", key="sel_fashion", use_container_width=True):
            st.session_state.modelo_sel = "fashion"
            st.session_state.resultados = None
            st.session_state.page = "classify"
            st.rerun()

    st.markdown(
        f'<div style="text-align:center;color:{MUTED};font-size:0.8rem;padding:0.5rem 0 0 0;">'
        f'Hacé clic en el modelo que querés usar. Podés cambiarlo en cualquier momento.</div>'
        f'</div>', unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════
#  PAGE: CLASSIFY
# ══════════════════════════════════════════════════

elif st.session_state.page == "classify":
    model_key = st.session_state.modelo_sel
    if model_key is None or model_key not in MODELOS:
        st.session_state.page = "select"
        st.rerun()

    info = MODELOS[model_key]
    modelo = cargar_modelo(info["archivo"])
    clases = info["clases"]

    # ── Top bar ──
    col_back, col_title, col_reset = st.columns([1, 5, 1])
    with col_back:
        if st.button("← Cambiar", key="btn_back"):
            st.session_state.page = "select"
            st.rerun()
    with col_title:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.7rem;padding:0.15rem 0;">'
            f'<span style="font-size:1.2rem;font-weight:700;color:{TEXT};">{info["icono"]} {info["nombre"]}</span>'
            f'<span style="font-size:0.75rem;background:rgba(56,189,248,0.1);color:{PRI};padding:0.15rem 0.7rem;'
            f'border-radius:100px;border:1px solid rgba(56,189,248,0.15);font-weight:500;">✓ Activo</span>'
            f'</div>', unsafe_allow_html=True
        )
    with col_reset:
        if st.button("🔄 Reset", key="btn_reset"):
            st.session_state.resultados = None
            st.rerun()

    # ── Upload card ──
    st.markdown(
        f'<div class="card">'
        f'<div class="card-title">📤 Subí imágenes</div>'
        f'<div class="card-sub">Subí una o varias imágenes en formato PNG o JPG. '
        f'El modelo las procesará automáticamente a 28×28 px en escala de grises.</div>',
        unsafe_allow_html=True
    )

    if modelo is None:
        st.error("🚫 No se encontró el archivo del modelo. Verificá que esté en la carpeta del proyecto.")
    else:
        uploaded_files = st.file_uploader(
            "PNG, JPG, JPEG — podés subir varias",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="uploader"
        )

        if uploaded_files:
            st.markdown(
                f'<div style="margin:0.3rem 0 0.5rem 0;color:{MUTED};font-size:0.85rem;">'
                f'{len(uploaded_files)} archivo(s) listo(s)</div>',
                unsafe_allow_html=True
            )

            accion, export = st.columns([1, 1])
            with accion:
                classify = st.button("🚀 Clasificar todas", key="btn_classify", use_container_width=True)
            with export:
                if st.session_state.get("resultados"):
                    csv = make_csv(st.session_state.resultados, clases)
                    st.download_button(
                        "📥 Exportar CSV", csv,
                        f"resultados_{model_key}.csv",
                        "text/csv", use_container_width=True
                    )

            if classify:
                resultados = []
                progreso = st.progress(0, text="Preparando...")
                total = len(uploaded_files)
                for idx, f in enumerate(uploaded_files):
                    progreso.progress((idx) / total, text=f"Analizando {f.name} ({idx+1}/{total})")
                    img = Image.open(f)
                    tensor, invertida, thumbnail_28 = preprocess_image(img)
                    preds = modelo.predict(tensor, verbose=0)
                    id_clase = np.argmax(preds[0])
                    prob = preds[0][id_clase] * 100
                    resultados.append({
                        "idx": idx + 1,
                        "nombre": f.name,
                        "imagen": img,
                        "thumbnail": thumbnail_28,
                        "clase": clases[id_clase],
                        "id_clase": int(id_clase),
                        "probabilidad": prob,
                        "invertida": invertida,
                        "predicciones": preds[0],
                    })
                progreso.progress(1.0, text="✅ Completado")
                progreso.empty()
                st.session_state.resultados = resultados
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Results ──
    resultados = st.session_state.get("resultados")
    if resultados:
        probs = [r["probabilidad"] for r in resultados]
        avg_conf = np.mean(probs)
        med_conf = np.median(probs)
        std_conf = np.std(probs)
        high = sum(1 for p in probs if p >= 80)
        mid = sum(1 for p in probs if 50 <= p < 80)
        low = sum(1 for p in probs if p < 50)

        # ── Dashboard KPIs ──
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">📊 Resultados generales</div>'
            f'<div class="card-sub">Resumen estadístico de la clasificación de {len(resultados)} imagen(es).</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="kpi-row">'
            f'<div class="kpi"><div class="nm">{len(resultados)}</div><div class="lb">Imágenes</div></div>'
            f'<div class="kpi"><div class="nm blue">{avg_conf:.1f}%</div><div class="lb">Promedio</div></div>'
            f'<div class="kpi"><div class="nm blue">{med_conf:.1f}%</div><div class="lb">Mediana</div></div>'
            f'<div class="kpi"><div class="nm">{std_conf:.1f}%</div><div class="lb">Desviación</div></div>'
            f'</div>'
            f'<div class="kpi-row">'
            f'<div class="kpi"><div class="nm green">{high}</div><div class="lb">Alta ≥80%</div></div>'
            f'<div class="kpi"><div class="nm yellow">{mid}</div><div class="lb">Media 50-80%</div></div>'
            f'<div class="kpi"><div class="nm red">{low}</div><div class="lb">Baja {"<"}50%</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        if avg_conf >= 80:
            st.markdown(f'<div class="msg ok">✅ Buen resultado — confianza promedio del {avg_conf:.1f}%</div>', unsafe_allow_html=True)
        elif avg_conf >= 50:
            st.markdown(f'<div class="msg info">📊 Confianza promedio del {avg_conf:.1f}% — revisá las imágenes con baja puntuación</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg warn">⚠️ Confianza promedio baja ({avg_conf:.1f}%). Probá con imágenes más nítidas.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Chart ──
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">📈 Comparativa</div>'
            f'<div class="card-sub">Cada barra representa una imagen. Color verde = alta confianza, amarillo = media, rojo = baja.</div>',
            unsafe_allow_html=True
        )
        chart_df = pd.DataFrame({
            "Archivo": [r["nombre"][:16] + ("…" if len(r["nombre"]) > 16 else "") for r in resultados],
            "Confianza": probs,
            "Nombre": [r["nombre"] for r in resultados],
        })
        comp = alt.Chart(chart_df).transform_calculate(
            lvl="datum.Confianza >= 80 ? 'Alta' : datum.Confianza >= 50 ? 'Media' : 'Baja'"
        ).mark_bar(size=28, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Archivo:N", sort=None, axis=alt.Axis(labelAngle=30, labelLimit=70, labelFontSize=10, labelColor=TEXT)),
            y=alt.Y("Confianza:Q", axis=alt.Axis(format='.1f', title=None, labelColor=TEXT, gridColor=BORDER)),
            color=alt.Color("lvl:N", scale=alt.Scale(domain=["Alta","Media","Baja"], range=[SUCCESS,WARN,DANGER]), legend=alt.Legend(title="Nivel", labelColor=TEXT, titleColor=MUTED)),
            tooltip=[alt.Tooltip("Nombre:N", title="Archivo"), alt.Tooltip("Confianza:Q", title="Confianza (%)", format='.1f')]
        ).properties(height=200).configure_view(strokeOpacity=0)
        st.altair_chart(comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Individual results ──
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">🔍 Resultados por imagen</div>'
            f'<div class="card-sub">Cada tarjeta muestra la imagen original, cómo la "ve" el modelo (28×28), '
            f'la categoría asignada, la confianza y las 3 clases más probables.</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(2)
        for i, r in enumerate(resultados):
            cl = conf_cls(r["probabilidad"])
            with cols[i % 2]:
                st.markdown(
                    f'<div class="res-card">'
                    f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
                    f'<span style="font-size:0.75rem;color:{MUTED};font-weight:500;">#{r["idx"]}</span>'
                    f'<span style="font-weight:600;color:{TEXT};font-size:0.9rem;">{r["nombre"]}</span>'
                    f'</div>'
                    f'<div style="display:flex;gap:0.8rem;">'
                    f'<div style="text-align:center;">'
                    f'<img src="data:image/png;base64,{img_to_b64(r["imagen"])}" width="80" style="border-radius:6px;border:1px solid {BORDER};cursor:pointer;" '
                    f'onclick="document.getElementById(\'lb-img\').src=this.src;document.getElementById(\'lb\').style.display=\'flex\';">'
                    f'<div class="rl" style="margin-top:0.2rem;">Original</div>'
                    f'</div>'
                    f'<div style="text-align:center;">'
                    f'<img src="data:image/png;base64,{img_to_b64(r["thumbnail"])}" width="80" style="border-radius:6px;border:1px solid {BORDER}55;image-rendering:pixelated;">'
                    f'<div class="rl" style="margin-top:0.2rem;">28×28</div>'
                    f'</div>'
                    f'<div style="flex:1;">'
                    f'<div class="rl">Categoría</div><div><span class="rcat">{r["clase"]}</span></div>'
                    f'<div style="margin-top:0.4rem;"><div class="rl">Confianza</div>'
                    f'<div class="rval {cl}">{r["probabilidad"]:.1f}%</div>'
                    f'<div class="rbar"><div class="rbar-fill {cl}" style="width:{r["probabilidad"]}%;"></div></div></div>'
                    f'</div></div>'
                    f'<div style="display:flex;flex-wrap:wrap;gap:0.25rem;margin-top:0.5rem;padding-top:0.4rem;border-top:1px solid {BORDER}55;">'
                    f'<span class="badge" title="La imagen se redimensionó a 28×28 píxeles">📐 28×28</span>'
                    f'<span class="badge" title="Se convirtió a escala de grises (1 canal)">🌑 Grises</span>'
                    f'<span class="badge" title="Se invirtieron los colores (fondo claro → oscuro)">{"🔄 Invertida" if r["invertida"] else "✅ Normal"}</span>'
                    f'<span class="badge" title="Índice numérico de la clase asignada">ID: {r["id_clase"]}</span>'
                    f'<span class="badge" title="Tamaño original de la imagen">{r["imagen"].width}×{r["imagen"].height}px</span>'
                    f'</div></div>', unsafe_allow_html=True
                )

                if r["probabilidad"] >= 90:
                    st.markdown(f'<div class="msg ok">✅ Muy seguro — "{r["clase"]}" con {r["probabilidad"]:.1f}%</div>', unsafe_allow_html=True)
                elif r["probabilidad"] >= 80:
                    st.markdown(f'<div class="msg ok">✅ Buena confianza — {r["probabilidad"]:.1f}%</div>', unsafe_allow_html=True)
                elif r["probabilidad"] >= 50:
                    st.markdown(f'<div class="msg info">⚠️ Con dudas ({r["probabilidad"]:.1f}%) — revisá la imagen</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="msg warn">❌ Baja confianza ({r["probabilidad"]:.1f}%) — probá con otra imagen</div>', unsafe_allow_html=True)

                # Top 3
                probs_list = [float(p) * 100 for p in r["predicciones"]]
                sorted_idx = np.argsort(probs_list)[::-1][:3]
                top3_df = pd.DataFrame({
                    "Clase": [clases[i] for i in sorted_idx],
                    "Probabilidad": [probs_list[i] for i in sorted_idx]
                })
                d = alt.Chart(top3_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=22).encode(
                    x=alt.X("Clase:N", sort=None, axis=alt.Axis(labelFontSize=10, labelColor=TEXT)),
                    y=alt.Y("Probabilidad:Q", axis=alt.Axis(format='.1f', title=None, labelColor=TEXT, gridColor=BORDER)),
                    color=alt.Color("Probabilidad:Q", scale=alt.Scale(domain=[0,100], range=[SURFACE2, PRI]), legend=None),
                    tooltip=[alt.Tooltip("Clase:N"), alt.Tooltip("Probabilidad:Q", format='.1f')]
                ).properties(height=130).configure_view(strokeOpacity=0)
                st.altair_chart(d, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Lightbox ──
        st.markdown(
            '<div id="lb" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.88);z-index:9999;'
            'align-items:center;justify-content:center;cursor:pointer;" '
            'onclick="this.style.display=\'none\';">'
            '<img id="lb-img" style="max-width:90vw;max-height:90vh;border-radius:12px;">'
            '</div>',
            unsafe_allow_html=True
        )

    elif modelo is not None:
        st.markdown(
            f'<div style="text-align:center;padding:2rem 1rem;color:{MUTED};">'
            f'📤 Subí imágenes y hacé clic en "Clasificar todas" para ver los resultados aquí</div>',
            unsafe_allow_html=True
        )
