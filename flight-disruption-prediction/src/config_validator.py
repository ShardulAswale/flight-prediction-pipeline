import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = [
    "OPENSKY_CLIENT_ID",
    "OPENSKY_CLIENT_SECRET",
]

def validate_environment(env_path: Path = None) -> None:
    """
    Validates that all required environment variables and secrets are present.
    Loads from .env if available. Fail-fast if missing.
    """
    if env_path and env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        # Fallback to default .env resolution
        load_dotenv()
    
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    
    if missing_vars:
        error_msg = (
            f"CRITICAL: Missing required environment variables: {', '.join(missing_vars)}.\n"
            "Please ensure these are set in your environment or in a .env file.\n"
            "Refer to .env.example for required keys."
        )
        logger.error(error_msg)
        sys.exit(1)  # Fail fast
        
    logger.info("Environment validation successful. All required secrets are present.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validate_environment(Path('..') / '.env')
