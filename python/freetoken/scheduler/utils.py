from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_embeds: torch.Tensor | None = None
    # input_ids with image spans replaced by per-image content hashes; prefix-cache
    # keys only (the model always consumes the real input_ids).
    cache_key_ids: torch.Tensor | None = None
    # Chunked multimodal prefill only: cumulative image-placeholder count over the prompt
    # (``prefill._mm_rows_before``), built at the first split and reused by every chunk.
    mm_rows_before: torch.Tensor | None = None
    # Requeued by a context-pool retraction: input_ids = original prompt + tokens generated so
    # far, max_tokens = the remaining budget. Its readmission is not a new prompt (no second
    # PromptAdmittedMsg); stop strings keep matching from the original prompt end.
    resumed: bool = False
    stop_prompt_len: int | None = None
    # Chunked prefill: cached_len at which queued short prompts last overtook this one
    # (prefill.py _SHORT_FIRST allows one overtaking pass per chunk).
    overtaken_at: int = -1

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
