"""Phase-1 placeholder for single-transition internal parallel tools.

The current implementation remains sequential for simplicity and stability.
This module is kept so that later versions can add row-wise or chunk-wise
parallel execution without changing the public solver API.
"""


def map_rows(func, iterable):
    return [func(item) for item in iterable]
