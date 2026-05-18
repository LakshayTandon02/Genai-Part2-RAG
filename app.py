# futuristic_rag_ui.py

import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Nebula AI",
    page_icon="🚀",
    layout="wide"
)

load_dotenv()

# -----------------------------------
# PARTICLE BACKGROUND
# -----------------------------------

particles_js = """
<!DOCTYPE html>
<html>
<head>

<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>

<style>

html, body {
    margin: 0;
    padding: 0;
    overflow: hidden;
    background: transparent;
}

#particles-js {
    position: fixed;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #1e1b4b
    );
    z-index: -1;
}

</style>

</head>

<body>

<div id="particles-js"></div>

<script>

particlesJS("particles-js", {

  "particles": {
    "number": {
      "value": 120
    },

    "color": {
      "value": ["#8b5cf6", "#06b6d4", "#3b82f6"]
    },

    "shape": {
      "type": "circle"
    },

    "opacity": {
      "value": 0.6,
      "random": true
    },

    "size": {
      "value": 4,
      "random": true
    },

    "line_linked": {
      "enable": true,
      "distance": 140,
      "color": "#6366f1",
      "opacity": 0.3,
      "width": 1
    },

    "move": {
      "enable": true,
      "speed": 2,
      "direction": "none",
      "random": true,
      "straight": false,
      "out_mode": "bounce"
    }
  },

  "interactivity": {

    "detect_on": "canvas",

    "events": {

      "onhover": {
        "enable": true,
        "mode": "bubble"
      },

      "onclick": {
        "enable": true,
        "mode": "repulse"
      }

    },

    "modes": {

      "bubble": {
        "distance": 250,
        "size": 12,
        "duration": 2,
        "opacity": 1,
        "speed": 3
      },

      "repulse": {
        "distance": 200,
        "duration": 0.4
      }
    }
  },

  "retina_detect": true
});

</script>

</body>
</html>
"""

components.html(particles_js, height=0)

# -----------------------------------
# ADVANCED CSS
# -----------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
}

/* APP */

.stApp {
    background: transparent;
    color: white;
}

/* HIDE STREAMLIT */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* TITLE */

.title {

    text-align: center;

    font-size: 82px;

    font-weight: 800;

    margin-top: 40px;

    background: linear-gradient(
        to right,
        #38bdf8,
        #8b5cf6,
        #ec4899
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    animation: glow 2s infinite alternate;
}

@keyframes glow {

    from {
        filter: drop-shadow(0px 0px 10px #8b5cf6);
    }

    to {
        filter: drop-shadow(0px 0px 30px #06b6d4);
    }
}

/* SUBTITLE */

.subtitle {

    text-align: center;

    font-size: 20px;

    color: #d1d5db;

    margin-bottom: 40px;
}

/* GLASS PANEL */

.glass {

    background: rgba(255,255,255,0.06);

    border: 1px solid rgba(255,255,255,0.1);

    backdrop-filter: blur(25px);

    border-radius: 30px;

    padding: 30px;

    box-shadow:
    0px 0px 25px rgba(99,102,241,0.25);

    transition: 0.4s;
}

.glass:hover {

    transform: translateY(-6px);

    box-shadow:
    0px 0px 35px rgba(139,92,246,0.4);
}

/* INPUT */

.stTextInput input {

    background: rgba(255,255,255,0.07) !important;

    color: white !important;

    border-radius: 18px !important;

    border: 1px solid rgba(255,255,255,0.15) !important;

    padding: 18px !important;

    font-size: 18px !important;
}

/* BUTTON */

.stButton button {

    background: linear-gradient(
        to right,
        #8b5cf6,
        #3b82f6
    );

    border: none;

    border-radius: 20px;

    color: white;

    font-weight: bold;

    font-size: 18px;

    padding: 15px;

    transition: 0.4s;

    width: 100%;
}

.stButton button:hover {

    transform: scale(1.04);

    box-shadow:
    0px 0px 25px #8b5cf6,
    0px 0px 50px #06b6d4;
}

/* RESPONSE */

.response {

    background: rgba(255,255,255,0.06);

    border-radius: 25px;

    padding: 30px;

    font-size: 18px;

    line-height: 1.9;

    border: 1px solid rgba(255,255,255,0.1);

    backdrop-filter: blur(25px);

    animation: fade 1s ease;
}

@keyframes fade {

    from {
        opacity: 0;
        transform: translateY(30px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

/* SIDEBAR */

section[data-testid="stSidebar"] {

    background: rgba(255,255,255,0.05);

    backdrop-filter: blur(25px);

    border-right: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TITLE
# -----------------------------------

st.markdown(
    '<div class="title">NEBULA AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">⚡ Futuristic 3D AI PDF Assistant</div>',
    unsafe_allow_html=True
)

# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:

    st.header("📚 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    process_btn = st.button("🚀 Create AI Brain")

# -----------------------------------
# SESSION
# -----------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -----------------------------------
# PROCESS PDF
# -----------------------------------

if uploaded_file and process_btn:

    with st.spinner("⚡ Building Neural Memory..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())

            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        st.session_state.vectorstore = vectorstore

        st.success("🔥 AI Brain Ready")

# -----------------------------------
# PROMPT
# -----------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY from the provided context."
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

# -----------------------------------
# MODEL
# -----------------------------------

model = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------------
# MAIN
# -----------------------------------

st.markdown('<div class="glass">', unsafe_allow_html=True)

question = st.text_input(
    "💬 Ask Your AI"
)

ask = st.button("✨ Generate")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# RESPONSE
# -----------------------------------

if ask:

    if st.session_state.vectorstore is None:

        st.warning("Upload a PDF first.")

    else:

        with st.spinner("🧠 AI Thinking..."):

            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )

            docs = retriever.invoke(question)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": question
            })

            response = model.invoke(final_prompt)

            st.markdown("## 🤖 Response")

            st.markdown(
                f"""
                <div class="response">
                {response.content}
                </div>
                """,
                unsafe_allow_html=True
            )