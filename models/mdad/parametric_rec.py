from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseParametricMDAD


class ParametricMDADRec(BaseParametricMDAD):
    score_name = "rec"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruction_heads = nn.ModuleList(
            [nn.Linear(self.hidden_dim, cardinality) for cardinality in self.cardinalities]
        )
        self.to(self.device)

    def _reconstruction_logits_from_encoded(self, encoded: torch.Tensor) -> list[torch.Tensor]:
        return [head(encoded) for head in self.reconstruction_heads]

    def _compute_training_loss(
        self,
        x_clean: torch.Tensor,
        masked: torch.Tensor,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        rec_logits = self._reconstruction_logits_from_encoded(encoded)
        _, n_features = x_clean.shape

        n_masked_total = masked.sum().clamp_min(1)
        rec_loss_sum = torch.tensor(0.0, device=self.device)

        for feature_idx in range(n_features):
            masked_feature = masked[:, feature_idx]
            if not masked_feature.any():
                continue

            rec_loss_sum = rec_loss_sum + F.cross_entropy(
                rec_logits[feature_idx][masked_feature],
                x_clean[:, feature_idx][masked_feature],
                reduction="sum",
            )
        return rec_loss_sum / n_masked_total

    def _score_from_encoded(
        self,
        x_clean: torch.Tensor,
        masked: torch.Tensor,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        rec_logits = self._reconstruction_logits_from_encoded(encoded)
        batch_size, n_features = x_clean.shape
        rec_scores = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

        for feature_idx in range(n_features):
            masked_feature = masked[:, feature_idx]
            if not masked_feature.any():
                continue

            log_probs = F.log_softmax(rec_logits[feature_idx], dim=1)
            true_log_probs = log_probs.gather(1, x_clean[:, feature_idx:feature_idx + 1]).squeeze(1)
            rec_scores = rec_scores + masked_feature.float() * (-true_log_probs)

        denom = masked.sum(dim=1).clamp_min(1).float()
        return rec_scores / denom
