from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

dimension = 384

index = faiss.IndexFlatL2(dimension)

stored_chunks = []


def store_chunks(chunks):

    embeddings = model.encode(chunks)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    index.add(embeddings)

    stored_chunks.extend(chunks)

    return len(chunks)


def search_chunks(query, top_k=3):

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:

        if idx < len(stored_chunks):

            results.append(
                stored_chunks[idx]
            )

    return results