from __future__ import annotations

import numpy as np
import torch

from .base import BaseBatchedNonParametricMDAD


class NonParametricMDADRec(BaseBatchedNonParametricMDAD):
    score_name = "rec"

    @torch.no_grad()
    def score_batch(
        self,
        x: np.ndarray | torch.Tensor,
        probe_times: tuple[int, ...],
        n_probe_views: int = 8,
        ref_chunk_size: int = 16384,
    ) -> torch.Tensor:
        self._require_reference()
        x_t = self._to_tensor(x, dtype=torch.int64)
        batch_size, _ = x_t.shape

        if len(probe_times) == 0:
            raise ValueError("probe_times must contain at least one probe time.")
        if n_probe_views < 1:
            raise ValueError("n_probe_views must be at least 1.")

        rec_scores = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        for probe_time in probe_times:
            t_probe = torch.full(
                (batch_size,),
                int(probe_time),
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(n_probe_views):
                x_tilde, masked, visible = self._mask_view_batch(x_t, t_probe)
                weights = self._soft_kernel_weights(
                    x_tilde=x_tilde,
                    visible=visible,
                    ref_chunk_size=ref_chunk_size,
                )
                rec_scores += self._reconstruction_score_batch(
                    x=x_t,
                    masked=masked,
                    weights=weights,
                )

        total_probe_views = len(probe_times) * n_probe_views
        return (rec_scores / total_probe_views).cpu()
