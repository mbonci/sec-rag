"""Local LLM generation via Apple MLX (Apple Silicon native, no PyTorch MPS)."""

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty

from src.rag_prompt import SYSTEM_PROMPT, build_context

# 4-bit quantized — ~1 GB download, runs on Apple Neural Engine
DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def load_model(model_id: str = DEFAULT_MODEL):
    """Load model and tokenizer via MLX. Call once at startup."""
    print(f"Loading model {model_id} ...")
    model, tokenizer = load(model_id)
    print("Model ready.")
    return tokenizer, model


def generate(
    query: str,
    chunks: list[dict],
    tokenizer,
    model,
    max_new_tokens: int = 512,
) -> str:
    """Build prompt from retrieved chunks and generate an answer."""
    context = build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=0.1)
    rep_penalty = make_repetition_penalty(penalty=1.2, context_size=30)
    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens,
                        sampler=sampler, logits_processors=[rep_penalty], verbose=False)
