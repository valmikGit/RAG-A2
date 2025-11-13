import os
import os
import uvicorn
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from typing import List

# Optional: use sentence-transformers for embedding function if available
try:
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )
except Exception:
    SentenceTransformerEmbeddingFunction = None

# --- Configuration ---
# Use persistent ChromaDB if available (mounted into container at /app/chroma_data)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")
CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "/app/chroma_data")
EMBED_MODEL = os.getenv("CHROMA_EMBED_MODEL", "all-MiniLM-L6-v2")

CHROMA_COLLECTION = None
CHROMA_CLIENT = None

try:
    # Try opening a persistent client (this will succeed if your notebook mounted DB into the folder)
    CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)

    # Create an embedding function if available
    embedding_function = None
    if SentenceTransformerEmbeddingFunction is not None:
        try:
            embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        except Exception as e:
            print(f"Warning: failed to initialize embedding function: {e}")

    # If the collection exists, get it; otherwise create with embedding function if possible
    existing = [c.name for c in CHROMA_CLIENT.list_collections()]
    if COLLECTION_NAME in existing:
        CHROMA_COLLECTION = CHROMA_CLIENT.get_collection(COLLECTION_NAME)
        print(f"[INFO] Opened existing Chroma collection: {COLLECTION_NAME}")
    else:
        # create collection; if embedding function is None, collection will default to text-only
        if embedding_function:
            CHROMA_COLLECTION = CHROMA_CLIENT.create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
        else:
            CHROMA_COLLECTION = CHROMA_CLIENT.create_collection(name=COLLECTION_NAME)
        print(f"[INFO] Created new Chroma collection: {COLLECTION_NAME} at {CHROMA_PERSIST_PATH}")

except Exception as e:
    # Fallback to in-memory client if persistent client not available
    print(f"Error initializing persistent ChromaDB at '{CHROMA_PERSIST_PATH}': {e}")
    try:
        CHROMA_CLIENT = chromadb.Client()
        CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(COLLECTION_NAME)
        print("[INFO] Falling back to in-memory Chroma client.")
    except Exception as e2:
        print(f"Error initializing in-memory ChromaDB: {e2}")
        CHROMA_COLLECTION = None


# --- API Clients ---
try:
    # Initialize Gemini client (will pick up GEMINI_API_KEY from env)
    GEMINI_CLIENT = genai.Client()
    GEMINI_RAG_MODEL = os.getenv("GEMINI_RAG_MODEL", "gemini-2.5-flash")
    TOP_K = int(os.getenv("TOP_K", 3))
except Exception:
    # This will be caught when the app starts if GEMINI_API_KEY is missing
    print("[CRITICAL] Failed to initialize Gemini Client. Check GEMINI_API_KEY environment variable.")
    GEMINI_CLIENT = None
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
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_RAG_MODEL,
            contents=prompt,
            config={"system_instruction": "Answer the user query strictly based on the provided context."}
        )
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