from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

# -----------------------
# 1. DOCUMENTS
# -----------------------
docs = [
    Document(
        page_content="Artificial Intelligence is the simulation of human intelligence in machines that are programmed to think and learn.",
        metadata={"source": "AI_book"}
    ),
    Document(
        page_content="Machine Learning is a subset of AI that enables systems to learn from data and improve from experience without being explicitly programmed.",
        metadata={"source": "ML_book"}
    ),
    Document(
        page_content="Deep Learning uses neural networks with many layers to model complex patterns in data such as images, text, and speech.",
        metadata={"source": "DL_book"}
    ),
    Document(
        page_content="Natural Language Processing helps computers understand, interpret, and generate human language effectively.",
        metadata={"source": "NLP_book"}
    ),
    Document(
        page_content="Retrieval Augmented Generation combines search (retrieval) with LLM generation to improve factual accuracy using external knowledge.",
        metadata={"source": "RAG_notes"}
    )
]

# -----------------------
# 2. FREE EMBEDDINGS (IMPORTANT FIX)
# -----------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------
# 3. CHROMA VECTOR STORE
# -----------------------
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# -----------------------
# 4. TEST QUERY

result = vectorstore.similarity_search("What is the difference between AI and ML?" , k = 2)
# ----------------------
for r in result:
    print(r.page_content)
    print(r.metadata)
    print("\n")

retriever = vectorstore.as_retriever()
docs = retriever.invoke("Explain the concept of rag in simple terms ")

for d in docs:
    print(d.page_content)
