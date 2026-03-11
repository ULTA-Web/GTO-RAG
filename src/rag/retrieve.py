import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "db")

# 模型配置
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "model")

# 简单检查路径
if os.path.exists(LOCAL_MODEL_PATH):
    EMBEDDING_MODEL_PATH = LOCAL_MODEL_PATH
else:
    EMBEDDING_MODEL_PATH = LOCAL_MODEL_PATH

def get_retriever():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Vector database not found at {DB_PATH}. Please run src/rag/ingest.py first.")

    # Must use the same embedding model as ingestion
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    return vectorstore

def query_text_rag(query, k=5):
    """
    查询文本RAG检索
    :param query: 用户的问题
    :param k: 返回的最相关片段数量
    :return: list of (Document, score) tuples. Score is L2 distance (lower is better).
    """
    vectorstore = get_retriever()

    # results returns a list of (Document, score) tuples
    # score is distance (lower is better for L2 distance which Chroma uses by default)
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results