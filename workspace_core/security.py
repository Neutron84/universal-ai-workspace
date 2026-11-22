"""
Security & Secrets Manager
Safely retrieves secrets from Google Colab userdata or environment variables.
Provides secure random API key generation and token masking.
"""

import os
import secrets
from typing import Dict, Optional


class SecurityManager:
    """Handles credentials, API keys, and Colab secret retrieval safely."""

    _cached_keys: Dict[str, str] = {}

    @classmethod
    def get_secret(cls, key_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieves secret prioritizing:
        1. Google Colab userdata (google.colab.userdata.get)
        2. OS Environment Variables (os.environ.get)
        3. In-memory cache
        4. Default value
        """
        if key_name in cls._cached_keys:
            return cls._cached_keys[key_name]

        # 1. Colab Secrets
        try:
            from google.colab import userdata
            val = userdata.get(key_name)
            if val:
                cls._cached_keys[key_name] = str(val).strip()
                return cls._cached_keys[key_name]
        except Exception:
            pass

        # 2. Environment Variable
        env_val = os.environ.get(key_name)
        if env_val:
            cls._cached_keys[key_name] = env_val.strip()
            return cls._cached_keys[key_name]

        return default

    @classmethod
    def set_cached_secret(cls, key_name: str, value: str) -> None:
        """Sets an in-memory secret override."""
        cls._cached_keys[key_name] = value.strip()
        os.environ[key_name] = value.strip()

    @classmethod
    def get_hf_token(cls) -> Optional[str]:
        """Returns HuggingFace token for gated repos and higher rate limits."""
        token = cls.get_secret("HF_TOKEN") or cls.get_secret("HUGGING_FACE_HUB_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        return token

    @classmethod
    def get_civitai_api_key(cls) -> Optional[str]:
        """Returns CivitAI API Key."""
        return cls.get_secret("CIVITAI_API_KEY")

    @classmethod
    def get_cf_tunnel_token(cls) -> Optional[str]:
        """Returns Cloudflare Tunnel token for persistent custom domains."""
        return cls.get_secret("CF_TUNNEL_TOKEN")

    @classmethod
    def get_ngrok_token(cls) -> Optional[str]:
        """Returns Ngrok Auth Token."""
        return cls.get_secret("NGROK_AUTHTOKEN")

    @classmethod
    def generate_api_key(cls, length: int = 32) -> str:
        """Generates a cryptographically strong random API key."""
        return f"sk-aiws-{secrets.token_urlsafe(length)}"

    @classmethod
    def get_or_create_workspace_api_key(cls) -> str:
        """
        Retrieves the workspace server API key from secrets/env,
        or generates a strong random key.
        """
        key = cls.get_secret("WORKSPACE_API_KEY")
        if not key:
            key = cls.generate_api_key()
            cls.set_cached_secret("WORKSPACE_API_KEY", key)
        return key

    @staticmethod
    def mask_token(token: Optional[str]) -> str:
        """Masks a token for safe UI display (e.g. 'sk-aiws...xyz')."""
        if not token:
            return "Not Set"
        if len(token) <= 8:
            return "****"
        return f"{token[:4]}...{token[-4:]}"


def get_security_manager() -> SecurityManager:
    return SecurityManager()
