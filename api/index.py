"""Vercel entry point: Vercel serves the ASGI object named `app` from this file."""

from app.web.vercel_app import create_dashboard_app

app = create_dashboard_app()
