"""Generate from a local HF artifact using its declared cache capability."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def generate_text(
    model_path: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    num_visits: int | None = None,
    device: str = "cpu",
) -> str:
    """Greedy local generation, including the decoded prompt in the returned text.

    BOS is explicit, matching prepared training documents. No prompt truncation,
    downloads, sampling defaults, or terminal-KV approximations are introduced.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path)
    if not path.is_dir() or max_new_tokens < 1:
        raise ValueError("provide a local model directory and positive max_new_tokens")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, local_files_only=True).to(device).eval()
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        tokens.insert(0, tokenizer.bos_token_id)
    if not tokens:
        raise ValueError("an empty prompt needs a tokenizer BOS token")
    if len(tokens) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus max_new_tokens exceeds the model context; use a shorter prompt or generation length")
    inputs = torch.tensor([tokens], dtype=torch.long, device=device)
    kwargs = {} if num_visits is None else {"num_visits": num_visits}
    with torch.inference_mode():
        generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=max_new_tokens,
                                   do_sample=False, use_cache=model.config.use_cache, **kwargs)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-visits", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    print(generate_text(args.model, args.prompt, max_new_tokens=args.max_new_tokens, num_visits=args.num_visits, device=args.device))


if __name__ == "__main__":
    main()
