import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback for stripped test envs
    def load_dotenv(*args, **kwargs):
        return False

# Load environment variables from config directory
load_dotenv('config/.env')
load_dotenv('config/.env.local', override=True)
