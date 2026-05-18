from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os

# Load env
load_dotenv()

# Load PDF
loader = PyPDFLoader("document loaders/Lakshay8sem PRoject file.pdf")
docs = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# FREE Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If answer is not present in context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

# FREE Groq Model
model = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    api_key=os.getenv("GROQ_API_KEY")
)

print("RAG System Ready")
print("Press 0 to exit")

while True:

    query = input("\nYou: ")

    if query == "0":
        break

    # Retrieve docs
    retrieved_docs = retriever.invoke(query)

    # Combine context
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # Final prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # LLM response
    response = model.invoke(final_prompt)

    print(f"\nAI: {response.content}")