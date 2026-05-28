from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


class BaseMDADScorer(ABC):
    score_name: str = "score"

    @abstractmethod
    def score_batch(
        self,
        x: np.ndarray | torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def score_samples(
        self,
        x: np.ndarray | torch.Tensor,
        batch_size: int = 256,
        verbose: bool = True,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x, dtype=np.int64)

        n_samples = x_np.shape[0]
        scores = np.zeros(n_samples, dtype=np.float32)

        iterator = range(0, n_samples, batch_size)
        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    iterator,
                    total=math.ceil(n_samples / batch_size),
                    desc=f"Scoring {self.score_name}",
                )
            except Exception:
                pass

        for start in iterator:
            end = min(start + batch_size, n_samples)
            batch_scores = self.score_batch(x_np[start:end], **kwargs)
            scores[start:end] = batch_scores.detach().cpu().numpy()

        return scores


class BaseParametricMDAD(nn.Module, BaseMDADScorer):
    def __init__(
        self,
        cardinalities: list[int],
        mask_schedule: tuple[float, ...],
        d_model: int = 128,
        hidden_dim: int = 256,
        layers: int = 3,
        dropout: float = 0.1,
        eps: float = 1e-12,
        device: Optional[str] = None,
        seed: int = 0,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1.")

        self.cardinalities = list(cardinalities)
        self.d = len(cardinalities)
        self.T = len(mask_schedule) - 1
        self.eps = float(eps)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.device = resolve_device(device)

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

        self.register_buffer("mask_schedule_t", torch.tensor(mask_schedule, dtype=torch.float32))
        self.register_buffer("mask_codes", torch.tensor(self.cardinalities, dtype=torch.long))

        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, d_model) for cardinality in self.cardinalities]
        )
        self.feature_id_embedding = nn.Embedding(self.d, d_model)
        self.time_embedding = nn.Embedding(self.T + 1, d_model)

        input_dim = self.d * d_model + d_model
        hidden_layers: list[nn.Module] = []
        for layer_idx in range(layers):
            hidden_layers.append(
                nn.Linear(input_dim if layer_idx == 0 else self.hidden_dim, self.hidden_dim)
            )
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(dropout))
        self.shared_backbone = nn.Sequential(*hidden_layers)

    def _to_tensor(
        self,
        x: np.ndarray | torch.Tensor,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(self.device, dtype=dtype, non_blocking=True)

    @torch.no_grad()
    def sample_masked_view(
        self,
        x: np.ndarray | torch.Tensor,
        t: np.ndarray | torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = self._to_tensor(x, dtype=torch.long)
        batch_size, n_features = x_t.shape
        if isinstance(t, int):
            t_t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t_t = self._to_tensor(t, dtype=torch.long).view(-1)

        p = self.mask_schedule_t[t_t]
        rand = torch.rand(
            (batch_size, n_features),
            generator=self.generator,
            dtype=torch.float32,
        ).to(self.device)
        masked = rand < p[:, None]
        visible = ~masked

        x_tilde = x_t.clone()
        mask_codes = self.mask_codes.unsqueeze(0).expand(batch_size, n_features)
        x_tilde[masked] = mask_codes[masked]
        return x_tilde, masked, visible

    def _embed_x_tilde(self, x_tilde: torch.Tensor) -> torch.Tensor:
        batch_size, n_features = x_tilde.shape
        if n_features != self.d:
            raise ValueError(f"Expected x_tilde.shape[1] = {self.d}, got {n_features}.")

        feature_ids = torch.arange(self.d, device=self.device)
        embeddings = []
        for feature_idx in range(self.d):
            feature_embedding = self.feature_embeddings[feature_idx](x_tilde[:, feature_idx])
            embeddings.append(
                feature_embedding + self.feature_id_embedding(feature_ids[feature_idx])
            )

        x_emb = torch.stack(embeddings, dim=1)
        return x_emb.reshape(batch_size, -1)

    def _encode_conditioned(
        self,
        x_tilde: np.ndarray | torch.Tensor,
        t: np.ndarray | torch.Tensor | int,
    ) -> torch.Tensor:
        x_tilde_t = self._to_tensor(x_tilde, dtype=torch.long)
        batch_size, _ = x_tilde_t.shape

        if isinstance(t, int):
            t_t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t_t = self._to_tensor(t, dtype=torch.long).view(-1)

        x_flat = self._embed_x_tilde(x_tilde_t)
        t_emb = self.time_embedding(t_t)
        return self.shared_backbone(torch.cat([x_flat, t_emb], dim=1))

    @abstractmethod
    def _compute_training_loss(
        self,
        x_clean: torch.Tensor,
        masked: torch.Tensor,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _score_from_encoded(
        self,
        x_clean: torch.Tensor,
        masked: torch.Tensor,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def fit(
        self,
        x_train: np.ndarray | torch.Tensor,
        epochs: int = 20,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> "BaseParametricMDAD":
        x_train_t = self._to_tensor(x_train, dtype=torch.long)
        n_samples = x_train_t.shape[0]
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        if verbose:
            try:
                from tqdm import trange

                epoch_iter = trange(epochs, desc=f"Training {self.score_name}")
            except Exception:
                epoch_iter = range(epochs)
        else:
            epoch_iter = range(epochs)

        for _ in epoch_iter:
            perm = torch.randperm(n_samples, device=self.device)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = perm[start:end]
                xb = x_train_t[idx]

                t = torch.randint(
                    low=1,
                    high=self.T + 1,
                    size=(xb.shape[0],),
                    device=self.device,
                )
                x_tilde, masked, _ = self.sample_masked_view(xb, t)
                encoded = self._encode_conditioned(x_tilde, t)
                loss = self._compute_training_loss(
                    x_clean=xb,
                    masked=masked,
                    encoded=encoded,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                n_batches += 1

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=total_loss / max(n_batches, 1))

        return self

    @torch.no_grad()
    def score_samples(
        self,
        x: np.ndarray | torch.Tensor,
        batch_size: int = 256,
        verbose: bool = True,
        **kwargs,
    ) -> np.ndarray:
        was_training = self.training
        self.eval()
        scores = super().score_samples(
            x=x,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )
        if was_training:
            self.train()
        return scores

    @torch.no_grad()
    def score_batch(
        self,
        x: np.ndarray | torch.Tensor,
        probe_times: tuple[int, ...] = (1, 2, 3, 4),
        n_probe_views: int = 8,
    ) -> torch.Tensor:
        x_t = self._to_tensor(x, dtype=torch.long)
        batch_size, _ = x_t.shape
        scores = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

        if len(probe_times) == 0:
            raise ValueError("probe_times must contain at least one probe time.")
        if n_probe_views < 1:
            raise ValueError("n_probe_views must be at least 1.")

        for probe_time in probe_times:
            t_probe = torch.full(
                (batch_size,),
                int(probe_time),
                dtype=torch.long,
                device=self.device,
            )
            for _ in range(n_probe_views):
                x_tilde, masked, _ = self.sample_masked_view(x_t, t_probe)
                encoded = self._encode_conditioned(x_tilde, t_probe)
                scores += self._score_from_encoded(
                    x_clean=x_t,
                    masked=masked,
                    encoded=encoded,
                )

        total_probe_views = len(probe_times) * n_probe_views
        return (scores / total_probe_views).cpu()


class BaseBatchedNonParametricMDAD(BaseMDADScorer):
    def __init__(
        self,
        cardinalities: list[int],
        mask_schedule: tuple[float, ...],
        lambda_kernel: float = 1.0,
        eps: float = 1e-12,
        device: Optional[str] = None,
        seed: int = 0,
    ):
        self.cardinalities = list(cardinalities)
        self.T = len(mask_schedule) - 1
        self.lambda_kernel = float(lambda_kernel)
        self.eps = float(eps)
        self.device = resolve_device(device)

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

        self.mask_schedule_t = torch.tensor(
            mask_schedule,
            dtype=torch.float32,
            device=self.device,
        )

        self.X_ref_: torch.Tensor | None = None
        self.cardinalities_t_: torch.Tensor | None = None
        self.mask_codes_: torch.Tensor | None = None

    def _to_tensor(
        self,
        x: np.ndarray | torch.Tensor,
        dtype: torch.dtype = torch.int64,
    ) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(self.device, dtype=dtype, non_blocking=True)

    def _require_reference(self) -> None:
        if self.X_ref_ is None or self.mask_codes_ is None or self.cardinalities_t_ is None:
            raise RuntimeError("Call fit() first.")

    def fit(self, x_ref: np.ndarray | torch.Tensor) -> "BaseBatchedNonParametricMDAD":
        x_ref_t = self._to_tensor(x_ref, dtype=torch.int64)
        if x_ref_t.ndim != 2:
            raise ValueError("X_ref must be 2D.")
        if x_ref_t.shape[1] != len(self.cardinalities):
            raise ValueError("Cardinalities length must equal number of features.")

        self.cardinalities_t_ = torch.tensor(
            self.cardinalities,
            dtype=torch.int64,
            device=self.device,
        )
        self.mask_codes_ = self.cardinalities_t_.clone()
        self.X_ref_ = x_ref_t

        for col_idx, cardinality in enumerate(self.cardinalities):
            col = self.X_ref_[:, col_idx]
            if torch.any(col < 0) or torch.any(col >= cardinality):
                raise ValueError(
                    f"Column {col_idx} has values outside [0, {cardinality - 1}]. "
                    "Reference data must contain only valid category codes."
                )

        return self

    def _mask_view_batch(
        self,
        x: torch.Tensor,
        t_probe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_reference()

        batch_size, n_features = x.shape
        p = self.mask_schedule_t[t_probe]

        rand = torch.rand(
            (batch_size, n_features),
            generator=self.generator,
            dtype=torch.float32,
        ).to(self.device)
        masked = rand < p[:, None]
        visible = ~masked

        x_tilde = x.clone()
        mask_codes = self.mask_codes_.unsqueeze(0).expand(batch_size, n_features)
        x_tilde[masked] = mask_codes[masked]
        return x_tilde, masked, visible

    def _soft_kernel_weights(
        self,
        x_tilde: torch.Tensor,
        visible: torch.Tensor,
        ref_chunk_size: int = 16384,
    ) -> torch.Tensor:
        self._require_reference()

        n_ref = self.X_ref_.shape[0]
        weights_chunks: list[torch.Tensor] = []

        for start in range(0, n_ref, ref_chunk_size):
            end = min(start + ref_chunk_size, n_ref)
            x_ref_chunk = self.X_ref_[start:end]
            mismatch = x_ref_chunk.unsqueeze(0) != x_tilde.unsqueeze(1)
            d_vis = (mismatch & visible.unsqueeze(1)).sum(dim=2).to(torch.float32)
            weights_chunks.append(torch.exp(-self.lambda_kernel * d_vis))

        return torch.cat(weights_chunks, dim=1)

    def _reconstruction_score_batch(
        self,
        x: torch.Tensor,
        masked: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        self._require_reference()

        batch_size, n_features = x.shape
        denom = weights.sum(dim=1).clamp_min(self.eps)
        total_score = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        for feature_idx in range(n_features):
            masked_feature = masked[:, feature_idx]
            if not torch.any(masked_feature):
                continue

            n_categories = self.cardinalities[feature_idx]
            true_vals = x[:, feature_idx]
            ref_vals = self.X_ref_[:, feature_idx]
            counts = torch.zeros(
                (batch_size, n_categories),
                dtype=torch.float32,
                device=self.device,
            )

            for category_idx in range(n_categories):
                mask_category = (ref_vals == category_idx).to(weights.dtype)
                counts[:, category_idx] = (weights * mask_category[None, :]).sum(dim=1)

            probs = (counts + self.eps) / (denom[:, None] + self.eps * n_categories)
            p_true = probs.gather(1, true_vals[:, None]).squeeze(1).clamp_min(self.eps)
            total_score = total_score + masked_feature.to(torch.float32) * (-torch.log(p_true))

        n_masked = masked.sum(dim=1).clamp_min(1).float()
        return total_score / n_masked
