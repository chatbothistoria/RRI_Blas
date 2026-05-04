import streamlit as st
import streamlit.components.v1 as components
import faiss
import json
import numpy as np
import os
import io
import re
from xml.sax.saxutils import escape
from sentence_transformers import SentenceTransformer
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ==============================================================
# 1. CONFIGURACIÓN DE PÁGINA Y PARÁMETROS GLOBALES
# ==============================================================
st.set_page_config(page_title="Asistente RRI - Blas Sierra", page_icon="🏫", layout="centered")

# ==============================================================
# 2. SISTEMA DE LOGIN (CONTRASEÑA)
# ==============================================================
if "autorizado" not in st.session_state:
    st.session_state.autorizado = False

if not st.session_state.autorizado:
    st.title("🔒 Acceso Restringido")
    st.write("Por favor, introduce la contraseña para acceder al Asistente del RRI.")
    
    # Caja de contraseña
    pwd = st.text_input("Contraseña:", type="password")
    
    if st.button("Entrar"):
        if pwd == "docenteblas":
            st.session_state.autorizado = True
            st.rerun() # Recarga la página para mostrar el chat
        else:
            st.error("⚠️ Contraseña incorrecta. Inténtalo de nuevo.")
            
    # El st.stop() hace que si no estás autorizado, la app se detenga aquí y no cargue el resto.
    st.stop() 

# --- A partir de aquí, el código solo se ejecuta si la contraseña es correcta ---

FETCH_CHUNKS = 15           
MAX_CHUNKS_TO_LLM = 6       
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2" 

SYSTEM_PROMPT = """Eres un experto analizando el Reglamento de Régimen Interno (RRI) del CEIP Blas Sierra.
Tu objetivo es responder a las dudas de los usuarios basándote ÚNICAMENTE en el contexto extraído de dicho documento.

REGLAS:
1. Lee detenidamente el contexto proporcionado. A veces la información está dividida en varios fragmentos.
2. Si el contexto contiene la respuesta (aunque sea de forma parcial o con sinónimos), redacta una respuesta clara, profesional y empática.
3. Cita las páginas en el texto de tu respuesta, pero NUNCA generes un apartado final de "Fuentes consultadas", bibliografía o referencias (el sistema lo añadirá automáticamente).
4. Si la información no está en el contexto, di educadamente: "Lo siento, pero no encuentro esa información exacta en el documento que tengo cargado."
5. NUNCA te inventes datos, normas o plazos que no estén en el texto.

CONTEXTO DE BÚSQUEDA:
{context}
"""

# ==============================================================
# 3. CARGA DE RECURSOS EN CACHÉ
# ==============================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_faiss_and_meta():
    bin_file = "faiss_documento.bin"
    json_file = "meta_documento.json"

    if not os.path.exists(bin_file) or not os.path.exists(json_file):
        return None, None

    index = faiss.read_index(bin_file)
    with open(json_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

embed_model = load_embedding_model()
client = load_groq_client()

# ==============================================================
# 4. MOTOR DE GENERACIÓN DE PDF
# ==============================================================
def generar_pdf(mensajes, titulo="Documento de Consulta"):
    """Genera un archivo PDF a partir de una lista de mensajes"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    estilo_normal = styles["Normal"]
    estilo_titulo = styles["Title"]
    
    flowables = [Paragraph(titulo, estilo_titulo), Spacer(1, 20)]
    
    for msg in mensajes:
        if msg["role"] == "system":
            continue
            
        rol = "USUARIO" if msg["role"] == "user" else "ASISTENTE RRI"
        
        texto = escape(msg["content"])
        texto = texto.encode('windows-1252', 'ignore').decode('windows-1252')
        texto = texto.replace('\n', '<br/>')
        texto = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', texto) 
        texto = re.sub(r'\*(.*?)\*', r'<i>\1</i>', texto)     
        
        flowables.append(Paragraph(f"<b>[{rol}]</b>", estilo_normal))
        flowables.append(Spacer(1, 5))
        flowables.append(Paragraph(texto, estilo_normal))
        flowables.append(Spacer(1, 15))
        
    doc.build(flowables)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ==============================================================
# 5. INTERFAZ DE USUARIO (UI)
# ==============================================================
st.title("🏫 Asistente RRI - CEIP Blas Sierra")

# Disclaimer estático
st.markdown("<p style='text-align: center; font-size: 15px; color: #888;'>⚠️ <i>Este asistente utiliza IA y puede cometer errores. Contrasta siempre la información con el documento oficial del RRI.</i></p>", unsafe_allow_html=True)

st.divider()

index, metadata = load_faiss_and_meta()

if index is None or metadata is None:
    st.error("⚠️ Faltan los archivos de la base de datos (faiss_documento.bin o meta_documento.json). Asegúrate de subirlos a GitHub.")
    st.stop()

# ==============================================================
# 6. GESTIÓN DE LA MEMORIA Y CHAT (LIENZO EN BLANCO)
# ==============================================================

# Empezamos con el chat totalmente en blanco para evitar el salto de pantalla
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Opciones de descarga solo en los mensajes del asistente
        if msg["role"] == "assistant": 
            msg_usuario = st.session_state.messages[i-1] if i>0 and st.session_state.messages[i-1]["role"] == "user" else {"role": "user", "content": "Consulta general"}
            
            mensajes_pdf_individual = [msg_usuario, msg]
            pdf_individual = generar_pdf(mensajes_pdf_individual, "Consulta RRI")
            pdf_historial = generar_pdf(st.session_state.messages, "Historial de Consultas RRI")
            
            col_espacio, col_btn_resp, col_btn_conv = st.columns([4, 3, 3])
            
            with col_btn_resp:
                st.download_button(
                    label="📥 Guardar respuesta",
                    data=pdf_individual,
                    file_name=f"consulta_rri_{i}.pdf",
                    mime="application/pdf",
                    key=f"dl_resp_{i}",
                    use_container_width=True
                )
                
            with col_btn_conv:
                st.download_button(
                    label="📄 Guardar conversación",
                    data=pdf_historial,
                    file_name="historial_rri.pdf",
                    mime="application/pdf",
                    key=f"dl_conv_{i}",
                    use_container_width=True
                )

# ==============================================================
# 7. LÓGICA DEL RAG (BÚSQUEDA Y CITAS)
# ==============================================================
def buscar_contexto(pregunta):
    vector = embed_model.encode([pregunta], convert_to_numpy=True).astype('float32')
    distancias, indices = index.search(vector, FETCH_CHUNKS)

    contexto_textos = []
    documentos_citados = set()

    for idx in indices[0]:
        if idx == -1 or idx >= len(metadata):
            continue
            
        meta = metadata[idx]
        texto = meta["chunk_text"]
        page = meta["page_num"]

        cita_formateada = f"- Página {page}"
        fragmento = f"--- [Página: {page}] ---\n{texto}\n"

        if fragmento not in contexto_textos:
            contexto_textos.append(fragmento)
            documentos_citados.add(cita_formateada)

        if len(contexto_textos) >= MAX_CHUNKS_TO_LLM:
            break

    # Ordenar las páginas citadas de menor a mayor
    citas_ordenadas = sorted(list(documentos_citados), key=lambda x: int(x.split(" ")[-1]))
    return "\n".join(contexto_textos), citas_ordenadas

# ==============================================================
# 8. INTERACCIÓN DEL USUARIO
# ==============================================================
if prompt := st.chat_input("Escribe tu pregunta sobre el Reglamento..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el RRI..."):
            contexto_str, citas = buscar_contexto(prompt)
            
        mensajes_api = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=contexto_str)}
        ]

        historial_previo = st.session_state.messages[:-1][-4:] 
        for m in historial_previo:
            contenido = m["content"]
            if m["role"] == "assistant" and "**📚 Páginas consultadas:**" in contenido:
                contenido = contenido.split("\n\n---")[0].strip()
                
            mensajes_api.append({"role": m["role"], "content": contenido})

        mensajes_api.append({"role": "user", "content": prompt})

        respuesta_placeholder = st.empty()
        respuesta_completa = ""

        try:
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=mensajes_api,
                temperature=0.1, 
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    respuesta_completa += chunk.choices[0].delta.content
                    respuesta_placeholder.markdown(respuesta_completa + "▌")

            if citas and "no encuentro esa información exacta" not in respuesta_completa.lower():
                pie_fuentes = "\n\n---\n**📚 Páginas consultadas:**\n" + "\n".join(citas)
                respuesta_completa += pie_fuentes

            respuesta_placeholder.markdown(respuesta_completa)

        except Exception as e:
            respuesta_completa = f"⚠️ Ocurrió un error al contactar con la IA: {e}"
            respuesta_placeholder.markdown(respuesta_completa)

        st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
        
        st.rerun()

# ==============================================================
# 9. BLOQUEO DE AUTOFOCUS DE STREAMLIT (TRUCO DE SCROLL)
# ==============================================================
if "scroll_inicial" not in st.session_state:
    components.html(
        """
        <script>
            const doc = window.parent.document;
            let counter = 0;
            
            function preventStreamlitAutofocus() {
                if (doc.activeElement) {
                    doc.activeElement.blur();
                }
                window.parent.scrollTo(0, 0);
                const containers = doc.querySelectorAll('.main, [data-testid="stAppViewContainer"]');
                containers.forEach(c => c.scrollTo(0, 0));
            }

            const intervalId = setInterval(() => {
                preventStreamlitAutofocus();
                counter++;
                if (counter >= 30) {
                    clearInterval(intervalId);
                }
            }, 50);
        </script>
        """,
        height=0, width=0
    )
    st.session_state.scroll_inicial = True