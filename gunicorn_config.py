"""
Gunicorn configuration for production deployment.
Usage: gunicorn -c gunicorn_config.py app:app
"""

bind = "0.0.0.0:5000"
workers = 1           # Single worker — shared camera state
threads = 4           # Multiple threads for concurrent requests
timeout = 120         # Long timeout for streaming video
worker_class = "gthread"
accesslog = "-"
errorlog = "-"
loglevel = "info"
