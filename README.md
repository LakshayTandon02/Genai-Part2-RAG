# 🚀 GenAI Part 2 — Advanced RAG System

An advanced **Retrieval-Augmented Generation (RAG)** application built using:

- ⚡ Groq Llama 3.1
- 🧠 LangChain
- 📚 ChromaDB
- 🤗 HuggingFace Embeddings
- 🔥 Streamlit UI
- 📄 PDF Knowledge Base

This project allows users to upload PDFs/books and chat with documents using AI-powered semantic search and retrieval.

---

# ✨ Features

- 📄 Upload PDFs / Books
- 🧠 AI-powered Question Answering
- ⚡ Fast Retrieval using ChromaDB
- 🤖 Groq Llama 3.1 Integration
- 🔍 Semantic Search using Embeddings
- 💬 Interactive Chat Interface
- 🌌 Futuristic Animated UI
- 🆓 Fully Free Stack

---

# 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| LangChain | RAG Pipeline |
| Groq | LLM Inference |
| HuggingFace | Embeddings |
| ChromaDB | Vector Database |
| Streamlit | Frontend UI |
| Python | Backend |

---

# 📂 Project Structure

```bash
Genai-Part2-RAG/
│
├── app.py
├── create_database.py
├── main.py
├── chroma_db/
├── document loaders/
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/LakshayTandon02/Genai-Part2-RAG.git
```

Move into the project directory:

```bash
cd Genai-Part2-RAG
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# 🚀 Run the Project

### Create Vector Database

```bash
python create_database.py
```

### Run Main RAG System

```bash
python main.py
```

### Run Streamlit UI

```bash
streamlit run app.py
```

---

# 🧠 How RAG Works

1. PDF is uploaded
2. Text is extracted
3. Documents are split into chunks
4. Embeddings are created
5. Chunks are stored in ChromaDB
6. User asks a question
7. Relevant chunks are retrieved
8. Groq LLM generates final response

---

# 📸 Future Improvements

- 🌐 Multi-PDF Support
- 🎤 Voice Assistant
- 🧾 OCR Support
- 🖼️ Image Understanding
- 📊 Chat History
- ☁️ Deployment
- 🤖 AI Agents Integration

---

# 👨‍💻 Author

Developed by **Lakshay Tandon**

---

# ⭐ Support

If you liked this project:

- ⭐ Star the repository
- 🍴 Fork the project
- 🚀 Share with others

---

# 📜 License

This project is open-source and available under the MIT License.
