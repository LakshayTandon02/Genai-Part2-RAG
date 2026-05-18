from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
loader = PyPDFLoader(r"document loaders\Lakshay8sem PRoject file.pdf")
docs = loader.load()

splitter = TokenTextSplitter(
    chunk_size=10000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print(len(chunks))