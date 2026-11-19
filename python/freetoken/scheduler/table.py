import torch

from freetoken.runtime_conf import runtime_overrides


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self._warned_cap = None
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        """Free slots admission may use: the boot slot count, lowered by the hot
        `max_running_requests` (runtime_conf.py) when set."""
        free = len(self._free_slots)
        cap = runtime_overrides().get("max_running_requests")
        if cap is None:
            return free
        if cap > self._max_running_reqs and cap != self._warned_cap:
            self._warned_cap = cap
            _logger().warning(f"max_running_requests={cap} above the boot slot count "
                              f"{self._max_running_reqs}; using {self._max_running_reqs}")
        used = self._max_running_reqs - free
        return max(0, min(free, cap - used))

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)

    def rebuild(self, page_table: torch.Tensor) -> None:
        """Re-point the page table, reallocate the token pool, and free all slots.

        Idle-only: all request slots are expected to be free at rebuild time.
        """
        self.page_table = page_table
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
        self._free_slots = list(range(self._max_running_reqs))


_LOGGER = None


def _logger():
    global _LOGGER
    if _LOGGER is None:
        from freetoken.utils import init_logger
        _LOGGER = init_logger("freetoken.scheduler.table")
    return _LOGGER
