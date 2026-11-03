"""Environment-driven settings. Secrets come from env only; nothing is hardcoded."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Telegram allows 1-256 chars of A-Z a-z 0-9 _ - in the webhook secret token.
_SECRET_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


class Settings(BaseSettings):
    # env_ignore_empty: `NAME=` in a .env file (as .env.example ships) means "not set"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    env: Literal["dev", "test", "prod"] = "dev"
    database_url: str = "sqlite+aiosqlite:///./bharatdesk.db"
    # Run `alembic upgrade head` at startup so a fresh clone just works. Turn off in
    # production if migrations are applied by the deploy pipeline instead.
    auto_migrate: bool = True

    telegram_bot_token: SecretStr | None = None
    telegram_mode: Literal["polling", "webhook"] = "polling"
    telegram_webhook_url: str | None = None
    telegram_webhook_secret: SecretStr | None = None
    # Only needed when the token is absent (e.g. printing links from scripts).
    telegram_bot_username: str | None = None

    # --- speech-to-text (Groq Whisper) and LLM (Gemini). Model names come from env only. ---
    groq_api_key: SecretStr | None = None
    whisper_model: str | None = None
    stt_min_avg_logprob: float = -0.8  # below this the transcript is read back, never trusted
    stt_max_no_speech_prob: float = 0.6
    low_confidence_flag: float = 0.9  # a field below this is flagged in the read-back
    stt_force_convert: bool = (
        False  # always ffmpeg -> wav before upload (default: only on rejection)
    )
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    nvidia_api_key: SecretStr | None = None
    nvidia_model: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_provider: Literal["gemini", "nvidia"] | None = None  # required only if both keys are set
    llm_timeout_seconds: int = 60  # per LLM request; a hang would otherwise freeze the queue

    # --- media limits ---
    media_dir: str = "./media"
    max_media_bytes: int = 20 * 1024 * 1024  # Telegram bots cannot download more than 20 MB
    max_pdf_bytes: int = 20 * 1024 * 1024  # uploaded bills/PDFs
    max_pdf_pages: int = 50  # per upload; each page is one model call
    bill_concurrency: int = 3  # pages read at once (provider rate limits are real)
    min_voice_ms: int = 1500
    max_voice_seconds: int = 120
    allowed_voice_mimes: tuple[str, ...] = ("audio/ogg", "audio/mpeg", "audio/mp4", "audio/webm")

    # --- conversation / approvals ---
    conversation_ttl_minutes: int = 30
    approval_ttl_hours: int = 24
    expiry_sweep_seconds: int = 60
    stranded_after_minutes: int = 10  # a note still 'processing' after this is given up on
    retention_sweep_hours: int = 24  # how often stored audio/photos past retention are deleted

    # --- cost estimates (integers; left None because prices change and must be set by you) ---
    stt_cost_paise_per_minute: int | None = None
    llm_cost_paise_per_1k_tokens: int | None = None

    sandbox_enabled: bool = True  # forced off when ENV=prod
    # Sweeps, sheet sync and retention purge run in the background on the real clock. Tests that
    # use a fake clock turn this off, or a sweep would expire their fake-dated approvals.
    background_tasks: bool = True

    # --- photos / slips ---
    max_photo_bytes: int = 10 * 1024 * 1024
    min_photo_dimension: int = 300  # px, shortest side as reported by Telegram
    slip_min_legibility: float = 0.5  # below this the rep is asked for a clearer photo

    # --- Google Sheets sync (service account) ---
    google_service_account_file: str | None = None
    google_service_account_json: SecretStr | None = None
    sync_interval_seconds: int = 15
    sync_max_attempts: int = 8
    sync_batch_size: int = 20

    # --- boss dashboard (signed, expiring link) ---
    dashboard_secret: SecretStr | None = None
    public_base_url: str | None = None
    dashboard_link_ttl_minutes: int = 60

    invite_ttl_hours: int = 24
    pending_action_ttl_minutes: int = 10
    consent_version: str = "v1"
    rate_limit_per_minute: int = 30

    @model_validator(mode="after")
    def _check_provider_models(self) -> Settings:
        if self.groq_api_key is not None and not self.whisper_model:
            raise ValueError("GROQ_API_KEY is set but WHISPER_MODEL is not (no hardcoded models)")
        if self.gemini_api_key is not None and not self.gemini_model:
            raise ValueError("GEMINI_API_KEY is set but GEMINI_MODEL is not (no hardcoded models)")
        if self.nvidia_api_key is not None and not self.nvidia_model:
            raise ValueError("NVIDIA_API_KEY is set but NVIDIA_MODEL is not (no hardcoded models)")
        if self.gemini_api_key is not None and self.nvidia_api_key is not None:
            if self.llm_provider is None:
                raise ValueError("both GEMINI_API_KEY and NVIDIA_API_KEY are set: set LLM_PROVIDER")
        if self.llm_provider == "nvidia" and self.nvidia_api_key is None:
            raise ValueError("LLM_PROVIDER=nvidia requires NVIDIA_API_KEY")
        if self.llm_provider == "gemini" and self.gemini_api_key is None:
            raise ValueError("LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        if self.env == "prod":
            self.sandbox_enabled = False
        if self.dashboard_secret is not None and len(self.dashboard_secret.get_secret_value()) < 32:
            raise ValueError("DASHBOARD_SECRET must be at least 32 characters")
        return self

    @model_validator(mode="after")
    def _check_webhook_mode(self) -> Settings:
        if self.telegram_mode != "webhook":
            return self
        if self.telegram_bot_token is None:
            raise ValueError("TELEGRAM_MODE=webhook requires TELEGRAM_BOT_TOKEN")
        if not self.telegram_webhook_url:
            raise ValueError("TELEGRAM_MODE=webhook requires TELEGRAM_WEBHOOK_URL")
        secret = self.telegram_webhook_secret
        if secret is None or not _SECRET_TOKEN_RE.match(secret.get_secret_value()):
            raise ValueError(
                "TELEGRAM_MODE=webhook requires TELEGRAM_WEBHOOK_SECRET "
                "(1-256 chars of A-Z a-z 0-9 _ -)"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
