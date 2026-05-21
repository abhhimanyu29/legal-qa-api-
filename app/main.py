from datetime import datetime
from typing import List
import shutil

from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session

from app.config import settings

from app.database import (
    engine,
    Base,
    get_db,
    SessionLocal
)

from app.models.document import Document

from app.schemas.document import (
    DocumentCreate,
    DocumentResponse
)

from app.services.chunk_service import chunk_text
from app.services.pdf_service import extract_text_from_pdf

from app.services.chroma_service import (
    store_chunks,
    search_chunks
)

from app.services.ai_service import (
    generate_answer
)

# Create database tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

# Conversation memory
conversation_memory = []


# Root endpoint
@app.get("/")
async def root():

    return {
        "status": "running"
    }


# Create document manually
@app.post(
    "/documents",
    response_model=DocumentResponse
)
def create_document(
    document: DocumentCreate,
    db: Session = Depends(get_db)
):

    new_document = Document(
        filename=document.filename
    )

    db.add(new_document)

    db.commit()

    db.refresh(new_document)

    return new_document


# Get all documents
@app.get(
    "/documents",
    response_model=List[DocumentResponse]
)
def get_documents(
    db: Session = Depends(get_db)
):

    documents = db.query(Document).all()

    return documents


# Upload PDF
@app.post(
    "/upload",
    response_model=DocumentResponse
)
async def upload_document(
    file: UploadFile = File(...)
):

    db = SessionLocal()

    # Save uploaded file
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Extract text
    extracted_text = extract_text_from_pdf(
        file_path
    )

    # Chunk text
    chunks = chunk_text(extracted_text)

    # Store chunks in ChromaDB
    stored_count = store_chunks(
        chunks,
        file.filename
    )

    print(
        f"Stored {stored_count} chunks in ChromaDB",
        flush=True
    )

    print(
        f"Total chunks created: {len(chunks)}",
        flush=True
    )

    # Save document metadata
    document = Document(
        filename=file.filename,
        upload_time=datetime.utcnow()
    )

    db.add(document)

    db.commit()

    db.refresh(document)

    return document


# Semantic search endpoint
@app.post("/search")
async def search(
    query: str
):

    results = search_chunks(query)

    return {
        "query": query,
        "results": results
    }


# Conversational AI Question Answering
@app.post("/ask")
async def ask(
    query: str
):

    # Store user message
    conversation_memory.append(
        {
            "role": "user",
            "content": query
        }
    )

    # Retrieve chunks
    results = search_chunks(query)

    # Build retrieval context
    retrieval_context = "\n".join(
        [r["chunk"] for r in results]
    )

    # Build conversation history
    history = "\n".join(
        [
            f"{m['role']}: {m['content']}"
            for m in conversation_memory[-6:]
        ]
    )

    # Combined context
    full_context = f"""
Conversation History:
{history}

Retrieved Legal Context:
{retrieval_context}
"""

    # Generate answer
    answer = generate_answer(
        query=query,
        context=full_context
    )

    # Store assistant response
    conversation_memory.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return {
        "query": query,
        "answer": answer,
        "sources": results
    }