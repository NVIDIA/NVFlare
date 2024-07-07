import timeit
from collections import defaultdict


class CollectTimeContext:

    def __init__(self):
        self.metrics = {'count': 0, 'time_taken': 0.0, 'error_count': 0}
        self.start_time = None

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = timeit.default_timer() - self.start_time

        if exc_type is not None:
            # An exception occurred
            self.metrics['error_count'] += 1
        else:
            # No exception occurred
            self.metrics['count'] += 1
            self.metrics['time_taken'] += elapsed_time

        # Return False to propagate the exception if there was one
        return False


