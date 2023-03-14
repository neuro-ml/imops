def discard_arg(idx: int):
    def inner(f):
        def wrapper(*args):
            nonlocal idx
            if idx < 0:
                idx = len(args) + idx
            return f(*args[:idx], *args[idx + 1 :])

        return wrapper

    return inner
