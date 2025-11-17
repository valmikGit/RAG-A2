# Backend Testing Guide

## Quick Start

1. **Start the backend:**

   ```bash
   cd RAG_app/backend
   python app.py
   ```

   The backend should start on `http://localhost:8000`

2. **Run tests:**

   ### Option 1: Python Test Script (Recommended)

   ```bash
   python test_endpoints.py
   ```

   ### Option 2: PowerShell (Windows)

   ```powershell
   .\test_endpoints.ps1
   ```

   ### Option 3: Bash/curl (Linux/Mac/WSL)

   ```bash
   bash test_endpoints.sh
   ```

## Available Endpoints

### 1. Health Check

**GET** `/health`

Returns system status and diagnostics.

**Example:**

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "ok": true,
  "details": {
    "chroma_present": true,
    "chroma_path": "../../VectorDB/chroma_Data_v5",
    "collection_name": "anlp_rag_collection",
    "collection_count": 126,
    "gemini_initialized": true,
    "gemini_model": "gemini-2.5-flash",
    "top_k": 3,
    "embed_model": "all-MiniLM-L6-v2",
    "sample_chunk_preview": "...",
    "sample_metadata": {...}
  }
}
```

**What to check:**

- ✅ `chroma_present: true` - Database is loaded
- ✅ `collection_count > 0` - Chunks are loaded
- ✅ `gemini_initialized: true` - LLM is ready
- ✅ `sample_chunk_preview` - Should NOT contain "ACT X. SC. Y" patterns
- ✅ `sample_metadata` - Should contain `act` and `scene` fields

---

### 2. List Collections

**GET** `/collections`

Shows all available ChromaDB collections.

**Example:**

```bash
curl http://localhost:8000/collections
```

**Response:**

```json
{
  "collections": [
    {
      "name": "anlp_rag_collection",
      "count": 126,
      "metadata": {...},
      "is_active": true
    }
  ],
  "active_collection": "anlp_rag_collection",
  "database_path": "../../VectorDB/chroma_Data_v5"
}
```

---

### 3. Query RAG System

**POST** `/query`

Ask questions and get RAG-based answers.

**Example:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"user_query": "What is the relationship between Brutus and Caesar?"}'
```

**Request Body:**

```json
{
  "user_query": "Your question here"
}
```

**Response:**

```json
{
  "answer": "Based on the context, Brutus and Caesar...",
  "contexts": [
    "Retrieved chunk 1...",
    "Retrieved chunk 2...",
    "Retrieved chunk 3..."
  ]
}
```

**What to check:**

- ✅ `contexts` should NOT contain "79 Julius Caesar ACT 3. SC. 1" patterns
- ✅ `contexts` should be clean dialogue/text only
- ✅ Metadata (Act/Scene) is stored separately (not visible in response)

---

## Troubleshooting

### Issue: "95 Julius Caesar ACT 3. SC. 1" still appears in contexts

**Possible causes:**

1. **Database not regenerated:**

   - You modified the cleaning code but didn't re-run the notebook
   - **Solution:** Re-run the notebook cells to regenerate `chroma_Data_v5`

2. **Backend pointing to old database:**

   - Check `.env` file or `app.py` for `CHROMA_PERSIST_PATH`
   - **Solution:** Make sure it points to `../../VectorDB/chroma_Data_v5`

3. **Cache issue:**
   - Backend loaded old collection into memory
   - **Solution:** Restart the backend (`Ctrl+C` then `python app.py`)

### Verification Steps:

1. **Check notebook output:**

   ```
   [INFO] Cleaning Statistics:
     Chunks with Act/Scene metadata: 82
     Average text reduction: 5.45%
   ```

2. **Check database files modified time:**

   ```bash
   ls -ltr VectorDB/chroma_Data_v5/
   ```

   Should show recent timestamps

3. **Check backend logs:**

   ```
   [SUCCESS] Opened collection: anlp_rag_collection (count: 126)
   ```

4. **Test health endpoint:**
   ```bash
   curl http://localhost:8000/health | grep sample_chunk_preview
   ```
   Should NOT show ACT/SCENE markers

---

## Manual Testing with Browser

Visit these URLs in your browser:

- Health: http://localhost:8000/health
- Collections: http://localhost:8000/collections
- API Docs (Swagger): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Expected Behavior

### ✅ CORRECT (after fix):

```json
{
  "contexts": [
    "She kneels. CAESAR He lifts her up. DECIUS CAESAR CALPHURNIA..."
  ]
}
```

### ❌ INCORRECT (before fix):

```json
{
  "contexts": [
    "95 Julius Caesar ACT 3. SC. 1 She kneels. CAESAR He lifts her up..."
  ]
}
```

---

## Environment Variables

Create a `.env` file in `RAG_app/backend/`:

```env
GEMINI_API_KEY=your_api_key_here
CHROMA_PERSIST_PATH=../../VectorDB/chroma_Data_v5
CHROMA_COLLECTION_NAME=anlp_rag_collection
CHROMA_EMBED_MODEL=all-MiniLM-L6-v2
GEMINI_RAG_MODEL=gemini-2.5-flash
TOP_K=3
```
