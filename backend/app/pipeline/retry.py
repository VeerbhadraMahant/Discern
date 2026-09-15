import logging
import time

from google.genai import errors

logger = logging.getLogger("discern.retry")


def call_with_retry(fn, *args, attempts: int = 3, base_delay: float = 1.5, **kwargs):
    """Retry a Gemini API call on transient server errors (e.g. 503 'high demand').

    Client errors (bad request, auth, etc.) are not retried - they won't succeed
    on a second try and should surface immediately.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except errors.ServerError as e:
            last_error = e
            logger.warning("Gemini call failed (attempt %d/%d): %s", attempt + 1, attempts, e)
            if attempt < attempts - 1:
                time.sleep(base_delay * (2**attempt))
    raise RuntimeError(
        f"Gemini is temporarily overloaded after {attempts} attempts. Please try again in a moment."
    ) from last_error
