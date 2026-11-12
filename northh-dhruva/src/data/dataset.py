import os
from typing import Dict, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


def build_binary_shard(token_ids: list[int], output_path: str) -> Dict[str, Union[int, str]]:
    """Serializes integer token IDs into a memory-mapped binary uint16 shard.

    Args:
        token_ids: List of integer token IDs.
        output_path: Destination file path for the .bin file.

    Returns:
        Dictionary containing metadata: 'num_tokens', 'file_size_bytes', and 'path'.

    Raises:
        AssertionError: If any token ID is outside the range [0, 65534].
    """
    for tid in token_ids:
        assert 0 <= tid < 65535, f"Token ID {tid} cannot fit in uint16 (must be in [0, 65534])"

    # Ensure target directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    arr = np.array(token_ids, dtype=np.uint16)
    arr.tofile(output_path)

    file_size_bytes = os.path.getsize(output_path)
    return {
        "num_tokens": len(token_ids),
        "file_size_bytes": file_size_bytes,
        "path": output_path,
    }


class MemmapPretrainingDataset(Dataset):
    """Memory-mapped binary pretraining dataset for Northh Dhruva.

    CRITICAL ALIGNMENT SPECIFICATION (Task 5):
    -------------------------------------------
    In `DhruvaForCausalLM.forward(input_ids, labels)`:
        shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits, shift_labels)

    Because `DhruvaForCausalLM.forward` ALREADY performs the autoregressive shift
    internally (shifting logits left by 1 and labels right by 1), this dataset provides
    identical chunk slices:
        x = chunk
        y = chunk.clone()
    This ensures that at sequence position `t`, the logit produced from `x[t]` is
    evaluated against `y[t+1] = chunk[t+1]` without double-shifting!

    Args:
        bin_path: Path to the binary `.bin` token file (dtype=uint16).
        seq_len: Pretraining sequence context length.
    """

    def __init__(self, bin_path: str, seq_len: int) -> None:
        self.bin_path = bin_path
        self.seq_len = seq_len
        # Open lazily in read-only mode using numpy.memmap
        self.data: np.memmap = np.memmap(bin_path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        """Returns the number of non-overlapping chunks available.

        Formula: (len(self.data) - 1) // self.seq_len
        """
        if len(self.data) <= self.seq_len:
            return 0
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns token tensors (x, y) of shape (seq_len,) with dtype torch.long."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        start = idx * self.seq_len
        end = start + self.seq_len

        # Read uint16 chunk and convert to torch.long (torch.int64)
        chunk_np = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk_np)
        y = x.clone()

        return x, y

