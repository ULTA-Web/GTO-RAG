import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 定义绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "db")

# 模型配置：强制使用用户指定的本地路径
# 用户路径: G:\new_projects\GTO-RAG\model
LOCAL_MODEL_PATH = r"G:\new_projects\GTO-RAG\model"

if os.path.exists(LOCAL_MODEL_PATH) and os.listdir(LOCAL_MODEL_PATH):
    EMBEDDING_MODEL_PATH = LOCAL_MODEL_PATH
    print(f"Using local model from: {EMBEDDING_MODEL_PATH}")
else:
    # 既然用户明确要求只用本地，这里如果找不到应该报错提示
    print(f"Warning: Local model directory '{LOCAL_MODEL_PATH}' does not exist or is empty.")
    print("Please ensure your model files are in 'G:\\new_projects\\GTO-RAG\\model'")
    # 仍然赋值以便后续报错
    EMBEDDING_MODEL_PATH = LOCAL_MODEL_PATH

def ingest_documents():
    # 1. Load Documents
    print(f"Loading PDF files from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory '{DATA_PATH}' not found.")
        os.makedirs(DATA_PATH)
        print(f"Created '{DATA_PATH}'. Please put your PDF files there and run this script again.")
        return

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print("No PDF documents found. Please ensure your PDFs are in the 'data' folder.")
        return

    print(f"Loaded {len(documents)} pages.")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 稍微减小一点 chunk size 以获取更精准的片段
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Embed and Store
    print(f"Initializing embeddings...")
    # 这一步第一次运行会下载模型，可能需要几分钟
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    print(f"Creating vector store at {DB_PATH}...")

    # 清理旧数据库以防冲突
    if os.path.exists(DB_PATH):
        print("Removing existing database to rebuild with new embeddings...")
        try:
            shutil.rmtree(DB_PATH)
        except Exception as e:
            print(f"Warning: Could not delete old DB folder ({e}). If you get errors, manually delete the 'db' folder.")

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=DB_PATH
        )
        print(f"Ingestion complete. Vector store saved to '{DB_PATH}'.")
    except Exception as e:
        print(f"Error creating vector store: {e}")

if __name__ == "__main__":
    ingest_documents()
