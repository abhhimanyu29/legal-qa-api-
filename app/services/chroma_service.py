from rank_bm25 import BM25Okapi
import chromadb

from sentence_transformers import (
    SentenceTransformer
)

# Embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Persistent Chroma client
client = chromadb.PersistentClient(
    path="chroma_db"
)

# Collection
collection = client.get_or_create_collection(
    name="legal_documents"
)

# Store all chunks for BM25
all_chunks = []


def store_chunks(
    chunks,
    filename
):

    global all_chunks

    # Store chunks for BM25
    all_chunks.extend(chunks)

    embeddings = model.encode(chunks)

    ids = [
        f"{filename}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": filename,
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )

    return len(chunks)


def search_chunks(query, k=3):

    # Semantic search
    query_embedding = model.encode([query])

    semantic_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )

    semantic_chunks = semantic_results["documents"][0]

    # BM25 keyword search
    tokenized_chunks = [
        chunk.split()
        for chunk in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_query = query.split()

    keyword_results = bm25.get_top_n(
        tokenized_query,
        all_chunks,
        n=k
    )

    # Merge semantic + keyword results
    combined = list(
        set(
            semantic_chunks + keyword_results
        )
    )

    retrieved_chunks = []

    for chunk in combined:

        retrieved_chunks.append({
            "chunk": chunk
        })

    return retrieved_chunks