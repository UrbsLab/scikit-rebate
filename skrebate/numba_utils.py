import numpy as np
from numba import config

def resolve_num_threads(n_jobs) -> int:
    """Translate an ``n_jobs`` argument into a valid Numba thread count.

    ``-1`` means every thread in the configured Numba pool
    (``NUMBA_NUM_THREADS``), not ``NUMBA_DEFAULT_NUM_THREADS``.  Explicit values
    must be integers within ``[1, NUMBA_NUM_THREADS]``.
    """
    max_threads = int(config.NUMBA_NUM_THREADS)
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)):
        raise TypeError(f"n_jobs must be an integer, got {n_jobs!r}.")
    if n_jobs == -1:
        return max_threads
    if not 1 <= n_jobs <= max_threads:
        raise ValueError(
            f"n_jobs must be -1 or between 1 and {max_threads} (the configured Numba thread pool); got {n_jobs}."
        )
    return int(n_jobs)