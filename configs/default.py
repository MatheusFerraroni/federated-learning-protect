"""Application configuration and environment variable loading."""

import os
from src.utils.logging_utils import get_logger
from pathlib import Path
from dotenv import load_dotenv


logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

env_path = BASE_DIR / '.env'

if env_path.exists():
    load_dotenv(env_path)
    logger.info('Loaded .env', env_file=env_path)
else:
    logger.info('.env file not found, using system environment variables')


class Settings:  # pylint: disable=too-few-public-methods
    """Central configuration loaded from environment variables."""

    PROJECT_ROOT = os.getenv('PROJECT_ROOT', '/content/')
    DRIVE_ROOT = os.getenv('DRIVE_ROOT', 'None')


settings = Settings()

if settings.DRIVE_ROOT.lower() == 'none':
    settings.DRIVE_ROOT = None
