"""Run independent chunks of a loop over samples in worker processes.

``map_chunks`` splits ``range(n)`` into contiguous ``(start, stop)`` chunks, calls
``func(start, stop)`` on each and returns the results in chunk order, so joining them
gives exactly what one call ``func(0, n)`` would. With one worker (or where ``fork`` is
unavailable, i.e. Windows) it makes that single call and nothing is parallel.

Workers are forked, so ``func`` may be defined in a script and read that script's
module-level arrays without pickling them. The input scripts have no
``if __name__ == "__main__"`` guard, which is why a spawned pool is not used: it would
re-run the whole script in every worker.
"""

import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def n_workers():
    """Worker count from ``WORKERS`` (default 1), capped at the available cores."""
    requested = int(os.getenv("WORKERS", "1"))
    return max(1, min(requested, multiprocessing.cpu_count()))


def map_chunks(func, n, workers=1, progress=False, chunks_per_worker=4):
    """Return ``[func(start, stop), ...]`` over contiguous chunks of ``range(n)``."""
    if n <= 0:
        return []
    if workers > 1 and "fork" not in multiprocessing.get_all_start_methods():
        warnings.warn("fork is not available here, running on one worker")
        workers = 1
    if workers <= 1:
        return [func(0, n)]

    size = max(1, -(-n // (workers * chunks_per_worker)))
    bounds = [(start, min(start + size, n)) for start in range(0, n, size)]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(func, start, stop) for start, stop in bounds]
        return [future.result() for future in tqdm(futures, disable=not progress)]
