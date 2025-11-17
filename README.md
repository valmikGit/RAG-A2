# RAG System for Julius Caesar (Folger Shakespeare)

## About

This is a Retrieval-Augmented Generation (RAG) system built for querying Shakespeare's "Julius Caesar" text from the Folger Shakespeare Library edition. The system uses ChromaDB as a vector database with semantic chunking strategies to provide accurate, context-aware answers to questions about the play.

### Use Case

The RAG application allows users to:
- Ask natural language questions about Julius Caesar's plot, characters, and themes
- Get accurate answers with source references (Act, Scene, Page metadata)
- View retrieved chunks alongside their metadata in a clean two-column interface
- Leverage semantic search powered by embeddings for better context retrieval

## Running the RAG Application

### Prerequisites

- Anaconda or Miniconda installed
- Docker and Docker Compose installed

### 1. Environment Setup

The project uses two separate conda environments:

#### Main RAG Environment
Create the main environment using `environment.yaml`:

```bash
conda env create -f environment.yaml
```

This creates an environment with all dependencies needed for the RAG system, including:
- ChromaDB for vector storage
- LangChain for text processing
- Sentence Transformers for embeddings
- FastAPI for backend API
- Streamlit for frontend UI

#### RAGAS Evaluation Environment
Create the evaluation environment using `ragasenv.yaml`:

```bash
conda env create -f ragasenv.yaml
```

This creates a separate environment for running RAGAS evaluation metrics.

### 2. Running with Docker (Recommended)

The RAG application is designed to run entirely with Docker:

```bash
cd RAG_app
docker-compose up --build
```

This will:
- Build and start the FastAPI backend on port 8000
- Build and start the Streamlit frontend on port 8501
- Mount the ChromaDB vector database from `../VectorDB/chroma_Data_v_final`

**Access the application:**
- Frontend UI: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### 3. Environment Activation (For Development)

If you need to run notebooks or scripts locally:

**Activate main environment:**
```bash
conda activate anlp_rag
```

**Activate RAGAS environment:**
```bash
conda activate ragasenv
```

## Project Structure

```
RAG-A2/
├── RAG_app/                    # Docker-based application
│   ├── backend/                # FastAPI backend
│   ├── frontend/               # Streamlit frontend
│   └── docker-compose.yml      # Docker orchestration
├── VectorDB/                   # Vector database and notebooks
│   ├── chroma_Data_v_final/    # Cleaned ChromaDB database
│   └── RAGAS.ipynb             # Evaluation notebook
├── Chunking/                   # Text chunking notebooks
├── environment.yaml            # Main conda environment
└── ragasenv.yaml               # RAGAS evaluation environment
```

## Notes

- The system uses `all-MiniLM-L6-v2` embeddings for semantic search
- All Act/Scene/Page artifacts are cleaned from chunks but preserved as metadata
- The vector database contains 126 cleaned chunks from Julius Caesar
- Metadata is displayed separately in the frontend interface for better readability
