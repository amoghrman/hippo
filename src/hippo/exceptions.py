"""Public exceptions raised by Hippo."""

from __future__ import annotations

import uuid


class BatchPartialFailure(Exception):
    """Raised by ``remember_batch()`` when one or more chunks could not be committed.

    Chunks are atomic: either all items in a chunk succeed or the whole chunk is
    rolled back. Remaining chunks continue processing regardless of failures.

    Attributes:
        successful_ids: One entry per input item. Non-None where the item was
            successfully inserted; None where the item's chunk failed.
        failed_indices: Indices (into the original ``items`` list) that were not
            inserted due to a failed chunk.

    Example::

        try:
            ids = await memory.remember_batch(items)
        except BatchPartialFailure as e:
            print(f"Failed indices: {e.failed_indices}")
            print(f"Successful IDs: {[i for i in e.successful_ids if i is not None]}")
    """

    def __init__(
        self,
        successful_ids: list[uuid.UUID | None],
        failed_indices: list[int],
    ) -> None:
        self.successful_ids = successful_ids
        self.failed_indices = failed_indices
        n_failed = len(failed_indices)
        preview = failed_indices[:5]
        suffix = "..." if n_failed > 5 else ""
        super().__init__(
            f"remember_batch partially failed: {n_failed} item(s) at indices "
            f"{preview}{suffix} were not inserted."
        )
