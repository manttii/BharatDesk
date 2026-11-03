"""Vercel entry point for the boss dashboard.

Vercel looks for an ASGI object named `app` in standard files, and a root `index.py` is found
before `app/main.py`, which is the Telegram bot's own application and must never be the one
served here. Environment variables (DATABASE_URL, DASHBOARD_SECRET) are required: a missing one
raises a clear ConfigError at import time, which shows up in the function logs.
"""

from app.web.vercel_app import create_dashboard_app

app = create_dashboard_app()
