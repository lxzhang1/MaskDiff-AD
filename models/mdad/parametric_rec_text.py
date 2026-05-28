from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametricMDADRecText(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        mask_id: int,
        max_length: int,
        mask_schedule: tuple[float, ...],
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        device: str | None = None,
        seed: int = 0,
    ):
        super().__init__()
        if len(mask_schedule) < 2:
            raise ValueError("mask_schedule must contain at least two values.")

        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.mask_id = int(mask_id)
        self.max_length = int(max_length)
        self.mask_schedule = tuple(float(value) for value in mask_schedule)
        self.T = len(mask_schedule) - 1

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.time_embedding = nn.Embedding(self.T + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruction_head = nn.Linear(d_model, vocab_size)
        self.to(self.device)

    def _to_tensor(
        self,
        values: np.ndarray | torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)
        return values.to(self.device, dtype=dtype, non_blocking=True)

    @torch.no_grad()
    def sample_masked_view(
        self,
        input_ids: np.ndarray | torch.Tensor,
        attention_mask: np.ndarray | torch.Tensor,
        t: np.ndarray | torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids_t = self._to_tensor(input_ids, dtype=torch.long)
        attention_mask_t = self._to_tensor(attention_mask, dtype=torch.long)
        batch_size, sequence_length = input_ids_t.shape

        if isinstance(t, int):
            t_t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t_t = self._to_tensor(t, dtype=torch.long).view(-1)

        valid_pos = attention_mask_t.bool() & (input_ids_t != self.pad_id)
        probabilities = torch.tensor(
            self.mask_schedule,
            dtype=torch.float32,
            device=self.device,
        )[t_t]
        random_values = torch.rand(
            (batch_size, sequence_length),
            generator=self.generator,
            dtype=torch.float32,
        ).to(self.device)
        masked_pos = (random_values < probabilities[:, None]) & valid_pos

        for row in range(batch_size):
            if valid_pos[row].any() and not masked_pos[row].any():
                valid_indices = torch.nonzero(valid_pos[row], as_tuple=False).squeeze(1)
                choice = torch.randint(
                    0,
                    len(valid_indices),
                    (1,),
                    generator=self.generator,
                ).item()
                masked_pos[row, valid_indices[int(choice)]] = True

        x_tilde = input_ids_t.clone()
        x_tilde[masked_pos] = self.mask_id
        return x_tilde, masked_pos

    def forward_reconstruction(
        self,
        input_ids: np.ndarray | torch.Tensor,
        attention_mask: np.ndarray | torch.Tensor,
        t: np.ndarray | torch.Tensor | int,
    ) -> torch.Tensor:
        input_ids_t = self._to_tensor(input_ids, dtype=torch.long)
        attention_mask_t = self._to_tensor(attention_mask, dtype=torch.long)
        batch_size, sequence_length = input_ids_t.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"Expected sequence length <= {self.max_length}, got {sequence_length}."
            )

        if isinstance(t, int):
            t_t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t_t = self._to_tensor(t, dtype=torch.long).view(-1)

        position_ids = torch.arange(sequence_length, device=self.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, sequence_length)
        hidden = (
            self.token_embedding(input_ids_t)
            + self.position_embedding(position_ids)
            + self.time_embedding(t_t).unsqueeze(1)
        )
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask_t.bool())
        return self.reconstruction_head(hidden)

    def compute_loss(
        self,
        input_ids: np.ndarray | torch.Tensor,
        attention_mask: np.ndarray | torch.Tensor,
        t: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        input_ids_t = self._to_tensor(input_ids, dtype=torch.long)
        attention_mask_t = self._to_tensor(attention_mask, dtype=torch.long)
        t_t = self._to_tensor(t, dtype=torch.long).view(-1)
        x_tilde, masked_pos = self.sample_masked_view(input_ids_t, attention_mask_t, t_t)
        logits = self.forward_reconstruction(x_tilde, attention_mask_t, t_t)
        log_probs = F.log_softmax(logits, dim=-1)
        true_log_probs = log_probs.gather(2, input_ids_t.unsqueeze(-1)).squeeze(-1)
        token_nll = -true_log_probs * masked_pos.float()
        n_masked_total = masked_pos.sum().clamp_min(1).float()
        return token_nll.sum() / n_masked_total

    def fit_model(
        self,
        input_ids: np.ndarray | torch.Tensor,
        attention_mask: np.ndarray | torch.Tensor,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> "ParametricMDADRecText":
        input_ids_t = self._to_tensor(input_ids, dtype=torch.long)
        attention_mask_t = self._to_tensor(attention_mask, dtype=torch.long)
        n_samples = input_ids_t.shape[0]
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.train()

        if verbose:
            try:
                from tqdm import trange

                epoch_iter = trange(epochs, desc="Training text rec")
            except Exception:
                epoch_iter = range(epochs)
        else:
            epoch_iter = range(epochs)

        for _ in epoch_iter:
            permutation = torch.randperm(n_samples, device=self.device)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                indices = permutation[start:min(start + batch_size, n_samples)]
                batch_ids = input_ids_t[indices]
                batch_attention = attention_mask_t[indices]
                t = torch.randint(
                    1,
                    self.T + 1,
                    (batch_ids.shape[0],),
                    device=self.device,
                )
                loss = self.compute_loss(batch_ids, batch_attention, t)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                n_batches += 1

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=total_loss / max(n_batches, 1))

        return self

    @torch.no_grad()
    def score_batch(
        self,
        input_ids: np.ndarray | torch.Tensor,
        attention_mask: np.ndarray | torch.Tensor,
        probe_times: tuple[int, ...] = (1, 2, 3, 4),
        n_probe_views: int = 8,
    ) -> torch.Tensor:
        if n_probe_views < 1:
            raise ValueError("n_probe_views must be at least 1.")
        if not probe_times:
            raise ValueError("probe_times must contain at least one value.")
        if any(value < 1 or value > self.T for value in probe_times):
            raise ValueError(f"probe_times must be within [1, {self.T}].")

        input_ids_t = self._to_tensor(input_ids, dtype=torch.long)
        attention_mask_t = self._to_tensor(attention_mask, dtype=torch.long)
        batch_size = input_ids_t.shape[0]
        scores = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

        for probe_time in probe_times:
            t_probe = torch.full(
                (batch_size,),
                int(probe_time),
                dtype=torch.long,
                device=self.device,
            )
            for _ in range(n_probe_views):
                x_tilde, masked_pos = self.sample_masked_view(
                    input_ids_t,
                    attention_mask_t,
                    t_probe,
                )
                logits = self.forward_reconstruction(x_tilde, attention_mask_t, t_probe)
                log_probs = F.log_softmax(logits, dim=-1)
                true_log_probs = log_probs.gather(2, input_ids_t.unsqueeze(-1)).squeeze(-1)
                token_nll = -true_log_probs * masked_pos.float()
                denominator = masked_pos.sum(dim=1).clamp_min(1).float()
                scores += token_nll.sum(dim=1) / denominator

        total_probe_views = len(probe_times) * n_probe_views
        return (scores / total_probe_views).cpu()

    @torch.no_grad()
    def score_samples(
        self,
        input_ids: np.ndarray | torch.Tensor,
        attention_mask: np.ndarray | torch.Tensor,
        batch_size: int = 128,
        probe_times: tuple[int, ...] = (1, 2, 3, 4),
        n_probe_views: int = 8,
        verbose: bool = True,
    ) -> np.ndarray:
        input_array = (
            input_ids.detach().cpu().numpy()
            if isinstance(input_ids, torch.Tensor)
            else np.asarray(input_ids, dtype=np.int64)
        )
        attention_array = (
            attention_mask.detach().cpu().numpy()
            if isinstance(attention_mask, torch.Tensor)
            else np.asarray(attention_mask, dtype=np.int64)
        )
        scores = np.zeros(input_array.shape[0], dtype=np.float32)
        iterator = range(0, len(scores), batch_size)
        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    iterator,
                    total=math.ceil(len(scores) / batch_size),
                    desc="Scoring text rec",
                )
            except Exception:
                pass

        was_training = self.training
        self.eval()
        for start in iterator:
            end = min(start + batch_size, len(scores))
            scores[start:end] = self.score_batch(
                input_array[start:end],
                attention_array[start:end],
                probe_times=probe_times,
                n_probe_views=n_probe_views,
            ).numpy()
        if was_training:
            self.train()
        return scores
