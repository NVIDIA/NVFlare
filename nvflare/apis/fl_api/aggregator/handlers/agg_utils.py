def scale_dict(d, factor):
    if d is None:
        return None
    return {k: v * factor for k, v in d.items()}

def add_dict(a, b):
    if a is None or b is None:
        return a if a is not None else b
    return {k: a[k] + b[k] for k in a}

def average_dict(a, b):
    if a is None or b is None:
        return a if a is not None else b
    return {k: (a[k] + b[k]) / 2 for k in a}
