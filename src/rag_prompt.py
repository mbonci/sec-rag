"""Prompt templates for the RAG system."""

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the user's question using only the document excerpts provided below. "
    "Be concise and factual. Cite the company and section when relevant. "
    "If the excerpts do not contain enough information, say so."
)


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[{i}] {chunk.get('company_name', 'Unknown')} "
            f"({chunk.get('filing_year', '?')}) — {chunk.get('heading', '')}"
        )
        text = chunk.get("chunk_text", "")[:2000]
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks)
