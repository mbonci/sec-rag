"""Vector search over the FAISS index."""

import numpy as np


def search(
    query: str,
    index,
    metadata_df,
    embed_model,
    k: int = 5,
) -> list[dict]:
    """Embed query, search the FAISS index, return top-k rows as dicts."""
    vec = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    _, indices = index.search(vec, k)
    hits = indices[0]
    return metadata_df.iloc[hits].to_dict(orient="records")
