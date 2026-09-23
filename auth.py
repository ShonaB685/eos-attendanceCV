"""Minimal shared-secret gate for the /api/* routes. This is not real user
authentication -- no accounts, no sessions, no per-user permissions. It's a
floor, not a ceiling: stops random/automated hits on an exposed URL from
marking attendance or enrolling fake students, and is small/self-contained
enough that a deployment team can layer real auth (OAuth, JWT, whatever
fits their stack) on top without touching the recognition pipeline."""

import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY_FILE = os.path.join(BASE_DIR, ".api_key")


def get_api_key():
    """ATTENDANCE_API_KEY env var wins if set (what a real deployment
    should use). Otherwise falls back to a key in a local .api_key file,
    generating one on first run -- fine for local/dev use, not meant to be
    the production answer."""
    env_key = os.environ.get("ATTENDANCE_API_KEY")
    if env_key:
        return env_key

    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key

    key = secrets.token_urlsafe(32)
    with open(API_KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key)
    return key
