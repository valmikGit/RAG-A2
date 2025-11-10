import os
import uvicorn
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from typing import List

# --- Configuration ---
# NOTE: ChromaDB client setup - assumes a persistent local client or network endpoint
# For simplicity, this uses an in-memory client. Replace with your actual Chroma setup.
try:
    CHROMA_CLIENT = chromadb.Client()
    COLLECTION_NAME = "rag_collection"
    
    # Mock data setup: You MUST replace this with your actual Chroma initialization
    if COLLECTION_NAME not in [c.name for c in CHROMA_CLIENT.list_collections()]:
        CHROMA_COLLECTION = CHROMA_CLIENT.create_collection(COLLECTION_NAME)
        # Add a simple document for testing
        CHROMA_COLLECTION.add(
            documents=["The capital of France is Paris. It is famous for the Eiffel Tower.", 
                       "The fastest land animal is the cheetah.",
                       "Gemini models are powerful large language models developed by Google."],
            ids=["doc1", "doc2", "doc3"]
        )
    else:
        CHROMA_COLLECTION = CHROMA_CLIENT.get_collection(COLLECTION_NAME)

except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    # In a production app, you might want to stop here or use a fallback
    CHROMA_COLLECTION = None 


# --- API Clients ---
try:
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
        retrieved = CHROMA_COLLECTION.query(
            query_texts=[user_query],
            n_results=TOP_K,
            include=['documents']
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)