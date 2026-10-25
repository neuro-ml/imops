import os


def _max_num_threads(name: str) -> int:
    """The thread cap set by the environment variable `name`, 32 when it is unset or blank."""
    value = int(os.environ.get(name, '').strip() or 32)
    if value <= 0:
        raise ValueError(f'`{name}` must be positive, got {value}.')

    return value


ARGMAX_MAX_NUM_THREADS = _max_num_threads('IMOPS_ARGMAX_MAX_NUM_THREADS')
ROTATE_MAX_NUM_THREADS = _max_num_threads('IMOPS_ROTATE_MAX_NUM_THREADS')
