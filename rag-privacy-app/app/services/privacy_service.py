import base64
import hashlib
import os
import re
from typing import Dict

from cryptography.fernet import Fernet


def _build_fernet_key() -> bytes:
    """Build a deterministic Fernet key from env vars.

    Priority:
    1) PRIVACY_ENCRYPTION_KEY: user-provided secret
    2) SECRET_KEY: app secret
    3) static dev key fallback
    """
    raw_secret = (
        os.getenv("PRIVACY_ENCRYPTION_KEY")
        or os.getenv("SECRET_KEY")
        or "nexus-dev-privacy-key"
    )
    digest = hashlib.sha256(raw_secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


_FERNET = Fernet(_build_fernet_key())


def mask_sensitive_text(text: str) -> str:
    """Mask common sensitive identifiers to reduce leakage risk.

    Includes phone, email and Chinese ID card patterns.
    """
    if not text:
        return text

    masked = text

    # Mobile number: 13800138000 -> 138****8000
    masked = re.sub(r"\b(1\d{2})\d{4}(\d{4})\b", r"\1****\2", masked)

    # Email: user@example.com -> u***@example.com
    masked = re.sub(
        r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
        r"\1***\2",
        masked,
    )

    # Chinese ID card
    masked = re.sub(r"\b(\d{6})\d{8}(\w{4})\b", r"\1********\2", masked)

    return masked


def encrypt_text(plain_text: str) -> str:
    if plain_text is None:
        return ""
    return _FERNET.encrypt(plain_text.encode("utf-8")).decode("utf-8")


def decrypt_text(cipher_text: str) -> str:
    if not cipher_text:
        return ""
    return _FERNET.decrypt(cipher_text.encode("utf-8")).decode("utf-8")


def privacy_meta(raw_text: str, masked_text: str) -> Dict[str, int]:
    return {
        "raw_length": len(raw_text or ""),
        "masked_length": len(masked_text or ""),
        "masked_delta": len(raw_text or "") - len(masked_text or ""),
    }
