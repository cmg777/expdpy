"""Save/load ExPanD analysis configurations (JSON, optionally Fernet-encrypted)."""

from __future__ import annotations

import base64
import hashlib
import json

__all__ = ["dump_config", "load_config"]


def _derive_key(key_phrase: str) -> bytes:
    """Derive a urlsafe-base64 Fernet key from a passphrase via SHA-256."""
    return base64.urlsafe_b64encode(hashlib.sha256(key_phrase.encode()).digest())


def dump_config(config: dict, key_phrase: str | None = None) -> bytes:
    """Serialize ``config`` to JSON bytes, optionally encrypting with ``key_phrase``.

    Parameters
    ----------
    config
        The configuration mapping to save.
    key_phrase
        If given, the JSON is encrypted with an authenticated Fernet token derived from
        the phrase (tamper-evident; mirrors ExPanDaR's ``store_encrypted``).

    Returns
    -------
    bytes
        The (optionally encrypted) payload.
    """
    payload = json.dumps(config).encode()
    if key_phrase is None:
        return payload
    from cryptography.fernet import Fernet

    return Fernet(_derive_key(key_phrase)).encrypt(payload)


def load_config(raw: bytes, key_phrase: str | None = None) -> dict:
    """Load a configuration produced by :func:`dump_config`.

    Parameters
    ----------
    raw
        The payload bytes.
    key_phrase
        The passphrase used when saving (required if the payload is encrypted).

    Returns
    -------
    dict
        The configuration mapping.
    """
    if key_phrase is not None:
        from cryptography.fernet import Fernet

        raw = Fernet(_derive_key(key_phrase)).decrypt(raw)
    cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise ValueError("configuration payload is not a JSON object")
    return cfg
