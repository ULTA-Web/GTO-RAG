from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Define absolute paths for robustness
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db")

def get_rag_chain():
    # Check keys
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

    # 1. Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 2. Load Vector DB
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Vector database not found at {DB_PATH}. Please run src/ingest.py first.")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    # 3. Create Retriever
    # k=5 retrieves the 5 most relevant chunks
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 4. Create Prompt
    system_prompt = (
        "You are a Poker Strategy Expert specializing in GTO (Game Theory Optimal) play. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, say so. "
        "Keep answers concise, strategic, and data-driven where possible.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Build Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain
