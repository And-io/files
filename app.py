"""Detector de emociones con Streamlit.

Usa el modelo DistilBERT ajustado en clase (carpeta modelo_emociones_distilbert)
para clasificar una frase en seis emociones y mostrar sus probabilidades.
"""
import os
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
from transformers import pipeline

NOMBRE_CARPETA = "modelo_emociones_distilbert"
# La carpeta del modelo se busca junto a app.py, sin importar desde dónde se ejecute
RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), NOMBRE_CARPETA)
MAX_LEN = 64            # mismo valor usado en el entrenamiento
MAX_CARACTERES = 1000   # límite de la entrada para evitar textos enormes
MAX_FILAS_LOTE = 500    # límite de frases en el análisis por lotes

EJEMPLOS = [
    "I feel so happy and grateful today",
    "I am terrified of what might happen tomorrow",
    "I miss my grandmother so much, I can't stop crying",
    "I am furious, they lied to me again",
    "I love spending time with you",
    "I can't believe it, I won the lottery!",
    "Oh great, another Monday. Just what I needed.",  # sarcasmo: suele fallar
]

st.set_page_config(page_title="Detector de emociones", page_icon="🙂", layout="wide")


# ---------------------------------------------------------------- modelo
@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_clasificador(ruta):
    """Carga el modelo una sola vez; Streamlit lo reutiliza en cada clic."""
    return pipeline("text-classification", model=ruta)


def clasificar(clasificador, textos):
    """Devuelve, por frase, un diccionario {emoción: probabilidad}."""
    salidas = clasificador(
        textos, top_k=None, truncation=True, max_length=MAX_LEN, batch_size=16
    )
    return [{d["label"]: d["score"] for d in fila} for fila in salidas]


# ------------------------------------------------------------ utilidades
def detectar_idioma(texto):
    """Heurística simple: palabras muy frecuentes y signos propios del español."""
    t = " " + texto.lower() + " "
    if any(c in t for c in "¿¡ñáéíóú"):
        return "es"
    es = [" el ", " la ", " los ", " que ", " estoy ", " muy ", " porque ", " con ", " una ", " mi ", " es "]
    en = [" the ", " i ", " am ", " is ", " you ", " and ", " my ", " so ", " was ", " this ", " to "]
    p_es = sum(p in t for p in es)
    p_en = sum(p in t for p in en)
    if p_es > p_en:
        return "es"
    if p_en > 0:
        return "en"
    return "desconocido"


def grafico_probabilidades(probs):
    df = pd.DataFrame({"emoción": list(probs.keys()), "probabilidad": [v * 100 for v in probs.values()]})
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("probabilidad:Q", title="Probabilidad (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("emoción:N", sort="-x", title=None),
            tooltip=["emoción", alt.Tooltip("probabilidad:Q", format=".1f")],
        )
        .properties(height=260)
    )


def cargar_ejemplo(frase):
    """Callback de los botones de ejemplo: escribe la frase en el campo de texto."""
    st.session_state["texto"] = frase


def a_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# --------------------------------------------------------------- estado
st.session_state.setdefault("historial", [])
st.session_state.setdefault("ultimo", None)
st.session_state.setdefault("texto", "")

# ---------------------------------------------------------------- vista
st.title("🙂 Detector de emociones")
st.write(
    "Escribe una frase y el modelo DistilBERT ajustado en clase estimará si expresa "
    "tristeza, alegría, amor, ira, miedo o sorpresa."
)

# Validación: la carpeta del modelo debe existir y contener el modelo (config.json)
if not os.path.isfile(os.path.join(RUTA_MODELO, "config.json")):
    st.error(
        f"No se encontró el modelo en `{RUTA_MODELO}`. "
        f"Genera la carpeta `{NOMBRE_CARPETA}` con el Bloque 4.4 del notebook, "
        "colócala junto a `app.py` y verifica que contenga `config.json`."
    )
    st.stop()

clasificador = cargar_clasificador(RUTA_MODELO)

with st.sidebar:
    st.header("Configuración")
    umbral = st.slider(
        "Umbral de confianza (%)", 0, 100, 60,
        help="Si la emoción principal tiene una probabilidad menor, se mostrará una advertencia.",
    )
    st.divider()
    st.info(
        "🌐 **Idioma del modelo: inglés.** Fue ajustado con frases de Twitter en inglés "
        "(dataset dair-ai/emotion). Con frases en español los resultados pueden ser poco fiables."
    )

tab_uno, tab_lote, tab_hist = st.tabs(["Una frase", "Análisis por lotes", "Historial"])

# ------------------------------------------------------------ una frase
with tab_uno:
    st.caption("Frases de ejemplo:")
    columnas = st.columns(len(EJEMPLOS))
    for i, (col, frase) in enumerate(zip(columnas, EJEMPLOS)):
        col.button(f"Ejemplo {i + 1}", key=f"ej{i}", help=frase,
                   on_click=cargar_ejemplo, args=(frase,), use_container_width=True)

    st.text_area("Frase a analizar", key="texto", height=100,
                 placeholder="Ejemplo: I feel so happy today")

    if st.button("Analizar", type="primary"):
        texto = st.session_state["texto"].strip()
        if not texto:
            st.warning("Escribe una frase antes de analizar.")
        else:
            if len(texto) > MAX_CARACTERES:
                st.info(f"La frase supera {MAX_CARACTERES} caracteres; se recortó para analizarla.")
                texto = texto[:MAX_CARACTERES]
            with st.spinner("Analizando..."):
                probs = clasificar(clasificador, [texto])[0]
            principal = max(probs, key=probs.get)
            st.session_state["ultimo"] = {"texto": texto, "probs": probs, "principal": principal}
            st.session_state["historial"].append({
                "hora": datetime.now().strftime("%H:%M:%S"),
                "frase": texto,
                "emoción": principal,
                "confianza (%)": round(probs[principal] * 100, 1),
            })

    ultimo = st.session_state["ultimo"]
    if ultimo:
        probs, principal = ultimo["probs"], ultimo["principal"]
        confianza = probs[principal] * 100

        if detectar_idioma(ultimo["texto"]) == "es":
            st.warning("Parece una frase en español, pero el modelo fue entrenado en inglés. "
                       "Interpreta el resultado con cautela.")

        izq, der = st.columns([1, 2])
        izq.metric("Emoción principal", principal.capitalize(), f"{confianza:.1f} % de confianza",
                   delta_color="off")
        if confianza < umbral:
            izq.warning(f"Confianza baja: {confianza:.1f} % es menor que el umbral de {umbral} %.")
        der.altair_chart(grafico_probabilidades(probs), use_container_width=True)

# --------------------------------------------------------------- lotes
with tab_lote:
    st.write("Analiza varias frases a la vez: una por línea o desde un archivo CSV.")
    modo = st.radio("Origen de las frases", ["Escribir (una por línea)", "Subir CSV"], horizontal=True)

    frases = []
    if modo.startswith("Escribir"):
        bloque = st.text_area("Frases (una por línea)", height=150, key="bloque")
        frases = [l.strip() for l in bloque.splitlines() if l.strip()]
    else:
        archivo = st.file_uploader("Archivo CSV", type="csv")
        if archivo is not None:
            try:
                df_in = pd.read_csv(archivo)
                col = next((c for c in df_in.columns if c.lower() in ("text", "texto", "frase")),
                           df_in.columns[0])
                frases = df_in[col].dropna().astype(str).str.strip().tolist()
                frases = [f for f in frases if f]
                st.caption(f"Columna usada: `{col}` ({len(frases)} frases)")
            except Exception:
                st.error("No se pudo leer el CSV. Verifica que tenga una columna de texto.")

    if st.button("Analizar lote", type="primary"):
        if not frases:
            st.warning("No hay frases para analizar.")
        else:
            if len(frases) > MAX_FILAS_LOTE:
                st.info(f"Se analizarán solo las primeras {MAX_FILAS_LOTE} frases.")
                frases = frases[:MAX_FILAS_LOTE]
            frases = [f[:MAX_CARACTERES] for f in frases]
            with st.spinner("Analizando..."):
                resultados = clasificar(clasificador, frases)
            filas = []
            for frase, probs in zip(frases, resultados):
                principal = max(probs, key=probs.get)
                fila = {"frase": frase, "emoción": principal,
                        "confianza (%)": round(probs[principal] * 100, 1)}
                fila.update({e: round(p * 100, 1) for e, p in probs.items()})
                filas.append(fila)
            st.session_state["lote"] = pd.DataFrame(filas)

    if "lote" in st.session_state:
        df_lote = st.session_state["lote"].copy()
        df_lote.insert(3, "baja confianza", df_lote["confianza (%)"] < umbral)
        st.dataframe(df_lote, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Descargar resultados (CSV)", a_csv(df_lote),
                           file_name="resultados_emociones.csv", mime="text/csv")

# ------------------------------------------------------------- historial
with tab_hist:
    if not st.session_state["historial"]:
        st.info("Aún no has analizado ninguna frase en esta sesión.")
    else:
        df_hist = pd.DataFrame(st.session_state["historial"])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Descargar historial (CSV)", a_csv(df_hist),
                           file_name="historial_emociones.csv", mime="text/csv")
        if c2.button("🗑️ Borrar historial"):
            st.session_state["historial"] = []
            st.rerun()
