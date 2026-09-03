#!/usr/bin/env python3
"""Shared retry-with-backoff wrapper for Gemini API calls, used at every mode's
generate_content call site. Same reasoning as content_policy.py: one small piece of behavior
every mode needs, defined once rather than duplicated across nine call sites.

Exists because Gemini's own model-overloaded response (503 UNAVAILABLE, "This model is
currently experiencing high demand") is explicitly transient — Google's own error message says
to just try again — and empirically, a request that fails this way often succeeds on a same-day
retry a few seconds later (verified directly: 1 of 4 back-to-back identical calls succeeded
during a live demand spike). Retrying automatically, invisibly, means most users never see this
at all instead of getting a raw failure on bad luck alone.

Only retries error codes worth retrying: 503 (UNAVAILABLE) and 429 (RESOURCE_EXHAUSTED — a rate
limit that a short backoff can sometimes clear if the burst was brief). Every other error code
(400 bad request, 403 auth, a genuine content-safety block, etc.) fails immediately — retrying
those would just waste calls on a request that will never succeed differently.
"""

import time

from google.genai import errors as genai_errors

RETRYABLE_CODES = {429, 503}
# "Retry 2-3 times" per the spec: 1 initial attempt + 3 retries = 4 attempts total, the upper
# end of that range. Delays double each retry (1s, 2s, 4s) rather than being fixed, so a
# transient blip clears fast while a sustained outage doesn't hammer Gemini with tight retries.
MAX_ATTEMPTS = 4
BASE_DELAY_SECONDS = 1.0


def call_with_retry(fn, max_attempts=MAX_ATTEMPTS, base_delay=BASE_DELAY_SECONDS):
    """Calls fn() — a zero-arg callable wrapping one Gemini API call — retrying with exponential
    backoff on a retryable APIError. Re-raises immediately for a non-retryable error code, or
    once max_attempts is exhausted, so callers' existing except genai_errors.APIError handling
    (a fallback default, an HTTPException, etc.) still runs exactly as before — this only
    changes how many attempts happen before that handling sees the error."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except genai_errors.APIError as exc:
            if exc.code not in RETRYABLE_CODES or attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2**attempt))
