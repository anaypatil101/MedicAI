from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


# Step 1: Load raw PDF(s)
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)


# Step 3: Create Vector Embeddings
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Step 4: Store embeddings in FAISS
def build_vectorstore():
    documents = load_pdf_files(data=DATA_PATH)
    print(f"Loaded {len(documents)} PDF pages.")

    text_chunks = create_chunks(extracted_data=documents)
    print(f"Created {len(text_chunks)} text chunks.")

    embedding_model = get_embedding_model()
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store saved to {DB_FAISS_PATH}")


if __name__ == "__main__":
    build_vectorstore()