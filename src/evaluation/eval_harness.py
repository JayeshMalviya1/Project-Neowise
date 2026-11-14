"""Evaluation and autoregressive inference engine for Northh Dhruva."""

import math
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def compute_validation_perplexity(
    model: nn.Module,
    val_dataset: Union[Dataset, DataLoader],
    device: Union[str, torch.device] = "cpu",
    max_batches: int = 50,
) -> float:
    """Calculates cross-entropy loss and perplexity on held-out validation data.

    Args:
        model: DhruvaForCausalLM model instance.
        val_dataset: Validation Dataset or pre-constructed DataLoader.
        device: Device to run evaluation on.
        max_batches: Maximum number of batches to evaluate.

    Returns:
        Perplexity scalar: math.exp(min(20.0, avg_loss)).
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    if isinstance(val_dataset, DataLoader):
        val_loader = val_dataset
    else:
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (tuple, list)):
                x, y = (batch[0], batch[1]) if len(batch) >= 2 else (batch[0], batch[0])
            elif isinstance(batch, dict):
                x = batch.get("input_ids", batch.get("x"))
                y = batch.get("labels", batch.get("y", x))
            else:
                x, y = batch, batch

            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            outputs = model(x, labels=y)
            loss = outputs["loss"]
            if loss is not None:
                total_loss += loss.item()
                batch_count += 1

    if batch_count == 0:
        return 1.0

    avg_loss = total_loss / batch_count
    # Bound loss to prevent numerical overflow in exp
    bounded_loss = min(20.0, avg_loss)
    perplexity = math.exp(bounded_loss)
    return float(perplexity)


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
) -> int:
    """Samples next token ID using temperature scaling, top-k, and nucleus (top-p) filtering.

    Args:
        logits: Unnormalized logits tensor of shape (vocab_size,) or (1, vocab_size).
        temperature: Sampling temperature (<= 0.0 or < 1e-5 for greedy argmax).
        top_k: Highest probability tokens to keep for top-k filtering.
        top_p: Cumulative probability threshold for nucleus filtering.

    Returns:
        Sampled integer token ID.
    """
    logits = logits.squeeze().clone()

    # Greedy argmax if temperature is zero or non-positive
    if temperature <= 0.0 or temperature < 1e-5:
        return int(torch.argmax(logits, dim=-1).item())

    # 1. Temperature scaling
    logits = logits / max(temperature, 1e-5)

    # 2. Top-k filtering
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[-1]] = -float("Inf")

    # 3. Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right to retain the first token crossing the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    # 4. Softmax probability distribution and multinomial sampling
    probs = F.softmax(logits, dim=-1)

    # Fallback to greedy if distribution collapses
    if torch.isnan(probs).any() or probs.sum() <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def generate_text(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 40,
    device: Union[str, torch.device] = "cpu",
    top_p: float = 0.9,
    **kwargs: Any,
) -> str:
    """Generates autoregressive text continuation from a prompt string.

    Args:
        model: Trained DhruvaForCausalLM instance.
        tokenizer: ByteBPETokenizer instance.
        prompt: Initial prompt string.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering parameter.
        device: Device to run generation on.
        top_p: Top-p nucleus filtering parameter.

    Returns:
        Decoded text string including the prompt and generated continuation.
    """
    # Accommodate potential positional argument variations
    if isinstance(device, (int, float)):
        top_p, device = float(device), kwargs.get("device", "cpu")

    dev = torch.device(device)
    model.eval()
    model.to(dev)

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [getattr(tokenizer, "bos_token_id", 1)]

    eos_id = getattr(tokenizer, "eos_token_id", 2)
    curr_tokens = torch.tensor([token_ids], dtype=torch.long, device=dev)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if hasattr(model, "config") and hasattr(model.config, "max_seq_len"):
                max_len = model.config.max_seq_len
                input_tokens = curr_tokens[:, -max_len:]
            else:
                input_tokens = curr_tokens

            outputs = model(input_tokens)
            next_logits = outputs["logits"][0, -1, :]
            next_tid = sample_tokens(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if next_tid == eos_id:
                break

            next_tensor = torch.tensor([[next_tid]], dtype=torch.long, device=dev)
            curr_tokens = torch.cat([curr_tokens, next_tensor], dim=1)

    generated_ids = curr_tokens[0].tolist()
    return tokenizer.decode(generated_ids)


def stream_generate_tokens(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    device: Union[str, torch.device] = "cpu",
):
    """Yields generated text chunks in real-time as each token is sampled."""
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [getattr(tokenizer, "bos_token_id", 1)]

    eos_id = getattr(tokenizer, "eos_token_id", 2)
    curr_tokens = torch.tensor([token_ids], dtype=torch.long, device=dev)

    prev_text = tokenizer.decode(curr_tokens[0].tolist())

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if hasattr(model, "config") and hasattr(model.config, "max_seq_len"):
                max_len = model.config.max_seq_len
                input_tokens = curr_tokens[:, -max_len:]
            else:
                input_tokens = curr_tokens

            outputs = model(input_tokens)
            next_logits = outputs["logits"][0, -1, :]
            next_tid = sample_tokens(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if next_tid == eos_id:
                break

            next_tensor = torch.tensor([[next_tid]], dtype=torch.long, device=dev)
            curr_tokens = torch.cat([curr_tokens, next_tensor], dim=1)

            curr_text = tokenizer.decode(curr_tokens[0].tolist())
            delta = curr_text[len(prev_text):]
            prev_text = curr_text
            if delta:
                yield delta
