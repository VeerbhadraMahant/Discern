# From https://github.com/wyf0912/LLFlow code/utils/util.py (opt_get) at the pinned commit
# 115da161a96de868d67494a32db848e31f85bbc1, plus NoneDict from the same file's options module
# (CC BY-NC-SA 4.0, academic research only).


def opt_get(opt, keys, default=None):
    if opt is None:
        return default
    ret = opt
    for k in keys:
        ret = ret.get(k, None)
        if ret is None:
            return default
    return ret


class NoneDict(dict):
    """Config dict where a missing key reads as None, recursively, as the upstream code expects."""

    def __missing__(self, key):
        return None


def to_none_dict(value):
    if isinstance(value, dict):
        return NoneDict({k: to_none_dict(v) for k, v in value.items()})
    return value
