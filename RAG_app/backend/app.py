import os
import uvicorn
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optional: use sentence-transformers for embedding function if available
try:
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )
except Exception:
    SentenceTransformerEmbeddingFunction = None

# --- Configuration ---
# Use persistent ChromaDB from notebook's chroma_Data_v2 folder (READ-ONLY MODE)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "anlp_rag_collection")
# Point to the actual notebook database location
CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "../../VectorDB/chroma_Data_v2")
EMBED_MODEL = os.getenv("CHROMA_EMBED_MODEL", "all-MiniLM-L6-v2")

CHROMA_COLLECTION = None
CHROMA_CLIENT = None

try:
    # ✅ READ-ONLY: Open persistent client from notebook's database
    # Use Settings to ensure read-only access and prevent corruption
    settings = chromadb.Settings(
        allow_reset=False,
        anonymized_telemetry=False
    )
    CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH, settings=settings)
    print(f"[INFO] Opened ChromaDB in READ-ONLY mode from: {CHROMA_PERSIST_PATH}")

    # ✅ Create embedding function (must match notebook's embedding model)
    embedding_function = None
    if SentenceTransformerEmbeddingFunction is not None:
        try:
            embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
            print(f"[INFO] Initialized embedding function: {EMBED_MODEL}")
        except Exception as e:
            print(f"[WARN] Failed to initialize embedding function: {e}")

    # ✅ List existing collections
    existing = [c.name for c in CHROMA_CLIENT.list_collections()]
    print(f"[INFO] Collections found: {existing}")

    if not existing:
        print("[ERROR] No collections found! Make sure you've run the notebook to create the database first.")
        CHROMA_COLLECTION = None
    elif COLLECTION_NAME in existing:
        # ✅ CRITICAL: Use get_collection with embedding_function to avoid corruption
        if embedding_function:
            CHROMA_COLLECTION = CHROMA_CLIENT.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_function
            )
        else:
            CHROMA_COLLECTION = CHROMA_CLIENT.get_collection(name=COLLECTION_NAME)
        print(f"[SUCCESS] Opened collection: {COLLECTION_NAME} (count: {CHROMA_COLLECTION.count()})")
    else:
        # ✅ Use first available collection if exact name not found
        chosen = existing[0]
        if embedding_function:
            CHROMA_COLLECTION = CHROMA_CLIENT.get_collection(
                name=chosen,
                embedding_function=embedding_function
            )
        else:
            CHROMA_COLLECTION = CHROMA_CLIENT.get_collection(name=chosen)
        print(f"[INFO] Collection '{COLLECTION_NAME}' not found. Using '{chosen}' instead (count: {CHROMA_COLLECTION.count()})")
        COLLECTION_NAME = chosen

except Exception as e:
    print(f"[ERROR] Failed to initialize ChromaDB at '{CHROMA_PERSIST_PATH}': {e}")
    print("[ERROR] Make sure:")
    print("  1. You've run the notebook to create chroma_Data_v2")
    print("  2. The path is correct relative to backend/app.py")
    print("  3. The collection 'anlp_rag_collection' exists")
    CHROMA_COLLECTION = None


# --- API Clients ---
try:
    # Initialize Gemini client (load API key from .env)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables or .env file")
    
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_CLIENT = genai.GenerativeModel(os.getenv("GEMINI_RAG_MODEL", "gemini-2.5-flash"))
    GEMINI_RAG_MODEL = os.getenv("GEMINI_RAG_MODEL", "gemini-2.5-flash")
    TOP_K = int(os.getenv("TOP_K", 3))
    print(f"[SUCCESS] Gemini initialized with model: {GEMINI_RAG_MODEL}")
except Exception as e:
    # This will be caught when the app starts if GEMINI_API_KEY is missing
    print(f"[ERROR] Failed to initialize Gemini Client: {e}")
    print("[ERROR] Make sure GEMINI_API_KEY is set in .env file")
    GEMINI_CLIENT = None
    GEMINI_RAG_MODEL = None
# --- FastAPI Setup ---
app = FastAPI(title="Gemini RAG Backend")

# Request and Response Schemas
class QueryRequest(BaseModel):
    user_query: str
    
class RAGResponse(BaseModel):
    answer: str
    contexts: List[str]

# --- RAG Core Logic ---
def generate_rag_answer(user_query: str, contexts: List[str]) -> str:
    """Uses Gemini API to generate an answer based on context."""
    if not GEMINI_CLIENT:
        raise HTTPException(status_code=500, detail="Gemini Client is not initialized.")
        
    retrieved_context = "\n---\n".join(contexts)
    
    prompt = (
        f"You are a helpful assistant. Use ONLY the information provided in the context below to answer the user's query.\n\n"
        f"Context:\n{retrieved_context}\n\n"
        f"User Query:\n{user_query}\n\nAnswer:"
    )
    
    try:
        response = GEMINI_CLIENT.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API call failed: {e}")

@app.post("/query", response_model=RAGResponse)
async def handle_rag_query(request: QueryRequest):
    """Fetches contexts and generates a RAG-based answer."""
    user_query = request.user_query.strip()
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if not CHROMA_COLLECTION:
        raise HTTPException(status_code=503, detail="ChromaDB collection is unavailable.")

    # 1. Retrieve from ChromaDB
    try:
        # Use the collection query API; request documents and metadatas
        retrieved = CHROMA_COLLECTION.query(
            query_texts=[user_query],
            n_results=TOP_K,
            include=["documents", "metadatas"]
        )
        contexts = retrieved.get("documents", [[]])[0]
        
        if not contexts:
             return RAGResponse(
                answer="I could not find any relevant information in the knowledge base.",
                contexts=[]
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB retrieval failed: {e}")

    # 2. Generate Answer with Gemini
    answer = generate_rag_answer(user_query, contexts)
    
    return RAGResponse(
        answer=answer,
        contexts=contexts
    )


@app.get("/health")
def health_check():
    """Simple health endpoint."""
    ok = True
    details = {}
    details["chroma_present"] = CHROMA_COLLECTION is not None
    try:
        details["collection_count"] = CHROMA_COLLECTION.count() if CHROMA_COLLECTION else 0
    except Exception:
        details["collection_count"] = None
    details["gemini_initialized"] = GEMINI_CLIENT is not None
    return {"ok": ok, "details": details}


@app.get("/collections")
def list_collections():
    """Return basic info about collections available in the Chroma client."""
    try:
        cols = []
        if CHROMA_CLIENT:
            cols = [{"name": c.name, "metadata": getattr(c, 'metadata', None)} for c in CHROMA_CLIENT.list_collections()]
        return {"collections": cols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)