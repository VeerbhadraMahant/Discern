"""The single GPU-call decorator (system-design 8.1).

`gpu(duration=...)` wraps a function with `spaces.GPU` only when the `spaces` package is importable
and the `SPACE_ID` environment variable is set (that is, on a Hugging Face Space). Everywhere else
it only measures. A wrapped call is timed from the outside, so on ZeroGPU the figure includes time
spent waiting for a GPU slice: it is the wall time of the call, not kernel time.

If the first argument has a `record_gpu_ms(name, ms)` method, the measurement is reported to it,
but only on a runtime that has a GPU (a Space, or CUDA visible to torch).
"""

import functools
import importlib
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
SPACE_ENV_VAR = "SPACE_ID"
Duration = int | Callable[..., int]


def on_space() -> bool:
    """True on a Hugging Face Space with the `spaces` package installed."""
    if not os.environ.get(SPACE_ENV_VAR):
        return False
    try:
        importlib.import_module("spaces")
    except ImportError:
        return False
    return True


def gpu_available() -> bool:
    """Whether a call's wall time counts as GPU time: on a Space, or CUDA visible to torch."""
    if on_space():
        return True
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def gpu(duration: Duration) -> Callable[[F], F]:
    """Decorate a GPU-bound function. `duration` is the seconds requested from ZeroGPU, an int or
    a callable receiving the call's arguments; it comes from config, never a literal."""

    def decorate(fn: F) -> F:
        run: Callable[..., Any] = fn
        if on_space():
            spaces = importlib.import_module("spaces")
            run = spaces.GPU(duration=duration)(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return run(*args, **kwargs)
            finally:
                recorder = getattr(args[0], "record_gpu_ms", None) if args else None
                if callable(recorder) and gpu_available():
                    recorder(fn.__name__, (time.perf_counter() - started) * 1000.0)

        return wrapper  # type: ignore[return-value]

    return decorate
