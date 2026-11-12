import contextlib
import math
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DhruvaTrainer:
    """Pretraining Trainer for Northh Dhruva (Project BHARAT-LLM).

    Features:
    - Multi-device support: CUDA (with bfloat16/float16 AMP and optional torch.compile),
      Apple Silicon MPS, and CPU fallback.
    - Gradient accumulation for scalable effective batch sizes.
    - Decoupled weight decay parameter grouping (decay for 2D weights, no decay for 1D/biases).
    - Cosine learning rate decay with linear warmup.
    - Strict gradient clipping via norm.
    - Comprehensive checkpointing with resume capability and emergency exit handling.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_steps: int = 500,
        grad_accum_steps: int = 4,
        max_lr: float = 6e-4,
        min_lr: float = 6e-5,
        warmup_steps: int = 50,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "./checkpoints/dhruva_100m",
        save_interval: int = 100,
        log_interval: int = 10,
    ) -> None:
        self.raw_model: nn.Module = model
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.max_steps: int = max_steps
        self.grad_accum_steps: int = max(1, grad_accum_steps)
        self.max_lr: float = max_lr
        self.min_lr: float = min_lr
        self.warmup_steps: int = warmup_steps
        self.weight_decay: float = weight_decay
        self.grad_clip: float = grad_clip
        self.checkpoint_dir: str = checkpoint_dir
        self.save_interval: int = save_interval
        self.log_interval: int = log_interval

        # Setup compute device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = self._configure_optimizer()
        else:
            self.optimizer = optimizer

        # State tracking
        self.current_step: int = 0
        self.loss_history: List[float] = []

    def _get_base_model(self) -> nn.Module:
        """Returns the uncompiled base model."""
        if hasattr(self.model, "_orig_mod"):
            return self.model._orig_mod
        return self.model

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure AdamW with decoupled weight decay for 2D vs 1D parameters."""
        decay_params = []
        no_decay_params = []

        base_model = self._get_base_model()
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optim_groups,
            lr=self.max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def get_lr(self, step: int) -> float:
        """Computes learning rate for step according to cosine schedule with warmup."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.min_lr + (self.max_lr - self.min_lr) * (step / max(1, self.warmup_steps))
        if step >= self.max_steps:
            return self.min_lr
        # Cosine decay down to min_lr
        decay_ratio = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def _update_learning_rate(self, step: int) -> float:
        """Updates optimizer parameter groups with scheduled learning rate."""
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def save_checkpoint(self, path: Optional[str] = None, step: Optional[int] = None) -> str:
        """Saves current training state to disk."""
        target_step = self.current_step if step is None else step
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if path is None:
            save_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{target_step}.pt")
        else:
            save_path = path

        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        base_model = self._get_base_model()
        checkpoint_data: Dict[str, Any] = {
            "step": target_step,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "last_loss": self.loss_history[-1] if self.loss_history else None,
            "config": getattr(base_model, "config", None),
        }

        torch.save(checkpoint_data, save_path)

        # Also update last.pt symlink/copy if saved inside checkpoint_dir
        if path is None:
            last_path = os.path.join(self.checkpoint_dir, "last.pt")
            try:
                shutil.copyfile(save_path, last_path)
            except Exception:
                pass

        # Keep Apple Silicon unified memory pressure flat after checkpoint serialization
        if (
            self.device.type == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()

        return save_path

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """Loads training state from checkpoint file.

        Args:
            path: Explicit checkpoint file path. If None, looks for 'last.pt' in checkpoint_dir.

        Returns:
            True if checkpoint was found and loaded, False otherwise.
        """
        if path is None:
            checkpoint_file = os.path.join(self.checkpoint_dir, "last.pt")
        else:
            checkpoint_file = path

        if not os.path.isfile(checkpoint_file):
            return False

        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        base_model = self._get_base_model()
        base_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_step = checkpoint.get("step", 0)
        self.loss_history = checkpoint.get("loss_history", [])

        print(f"[Trainer] Resumed training state from '{checkpoint_file}' at step {self.current_step}")
        return True

    def fit(self, resume: bool = True) -> List[float]:
        """Main pretraining optimization loop.

        Args:
            resume: If True, attempts to restore checkpoint from checkpoint_dir.

        Returns:
            List of recorded losses for each optimization step.
        """
        if resume:
            self.load_checkpoint()

        self.model.train()
        data_iter = iter(self.train_loader)
        initial_step = self.current_step

        # Determine AMP context (skip GradScaler on MPS; fallback cleanly if PyTorch lacks MPS autocast)
        use_amp = False
        amp_dtype = torch.float32
        if self.device.type == "cuda":
            use_amp = True
            amp_dtype = (
                torch.bfloat16
                if (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
                else torch.float16
            )
        elif self.device.type == "mps":
            try:
                with torch.amp.autocast(device_type="mps", dtype=torch.float16):
                    pass
                use_amp = True
                amp_dtype = torch.float16
            except RuntimeError:
                use_amp = False
                amp_dtype = torch.float32

        amp_label = f"{self.device.type}:{amp_dtype}" if use_amp else f"{self.device.type}:float32"
        print(f"[Trainer] Starting pretraining on device '{self.device}' ({amp_label}):")
        print(f"          Steps: {self.current_step} -> {self.max_steps}")
        print(f"          Grad accumulation: {self.grad_accum_steps}")
        print(f"          Max LR: {self.max_lr:.2e}, Min LR: {self.min_lr:.2e}, Warmup: {self.warmup_steps}")

        start_time = time.time()
        try:
            while self.current_step < self.max_steps:
                step = self.current_step
                lr = self._update_learning_rate(step)
                self.optimizer.zero_grad(set_to_none=True)

                step_loss = 0.0
                for _ in range(self.grad_accum_steps):
                    try:
                        x, y = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_loader)
                        x, y = next(data_iter)

                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    amp_ctx = (
                        torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype)
                        if use_amp
                        else contextlib.nullcontext()
                    )
                    with amp_ctx:
                        outputs = self.model(x, labels=y)
                        loss = outputs["loss"] / self.grad_accum_steps

                    loss.backward()
                    step_loss += loss.item() * self.grad_accum_steps

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

                # Optimizer update
                self.optimizer.step()
                self.loss_history.append(step_loss)
                self.current_step += 1

                # Logging
                if (
                    self.current_step % self.log_interval == 0
                    or self.current_step == 1
                    or self.current_step == self.max_steps
                ):
                    elapsed = time.time() - start_time
                    steps_done = max(1, self.current_step - initial_step)
                    tokens_per_sec = (
                        self.train_loader.batch_size
                        * getattr(self.train_loader.dataset, "seq_len", 1)
                        * self.grad_accum_steps
                        * steps_done
                        / max(1e-4, elapsed)
                    )
                    norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    mem_suffix = ""
                    if (
                        self.device.type == "mps"
                        and hasattr(torch, "mps")
                        and hasattr(torch.mps, "current_allocated_memory")
                    ):
                        mps_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
                        mem_suffix = f" | MPS Mem: {mps_mb:.1f} MB"
                    print(
                        f"Step {self.current_step:4d}/{self.max_steps} | "
                        f"Loss: {step_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"GradNorm: {norm_val:.3f} | "
                        f"Tok/s: {tokens_per_sec:.0f}{mem_suffix}"
                    )

                # Periodic Checkpoint
                if self.current_step % self.save_interval == 0:
                    self.save_checkpoint()

            # Save final checkpoint
            self.save_checkpoint()
            print(f"[Trainer] Training completed successfully at step {self.current_step}!")

        except KeyboardInterrupt:
            print("\n[Trainer] KeyboardInterrupt received. Saving emergency checkpoint...")
            emergency_path = os.path.join(self.checkpoint_dir, "emergency_checkpoint.pt")
            self.save_checkpoint(path=emergency_path)
            print(f"[Trainer] Emergency checkpoint saved to '{emergency_path}'")
            raise

        return self.loss_history

