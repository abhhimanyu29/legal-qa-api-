from sentence_transformers import CrossEncoder

# Load reranker model
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_results(
    query,
    retrieved_chunks,
    top_k=3
):

    # Safety check
    if not retrieved_chunks:
        return []

    pairs = []

    valid_chunks = []

    for chunk in retrieved_chunks:

        if isinstance(chunk, dict) and "chunk" in chunk:

            pairs.append(
                (query, chunk["chunk"])
            )

            valid_chunks.append(chunk)

    # Another safety check
    if not pairs:
        return []

    scores = reranker.predict(pairs)

    ranked = []

    for chunk, score in zip(
        valid_chunks,
        scores
    ):

        ranked.append({

            "chunk": chunk["chunk"],

            "score": float(score)
        })

    ranked = sorted(
        ranked,
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked[:top_k]