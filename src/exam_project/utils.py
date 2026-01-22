from loguru import logger
import os


def validate_environment() -> None:
    """Validates that all needed environment variables are found in environment"""
    def validate_environment_var(environment_var: str) -> bool:
        env_var = os.getenv(environment_var)
        return all([env_var is not None, env_var != ""])
    
    logger.info("Asserting environment variables")

    assert validate_environment_var("MODEL_ARCHITECTURE"), 'Environment variable "MODEL_ARCHITECTURE" is None or ""'
    assert validate_environment_var("MODEL_NAME"), 'Environment variable "MODEL_NAME" is None or ""'
    assert validate_environment_var("WANDB_API_KEY"), 'Environment variable "WANDB_API_KEY" is None or ""'
    assert validate_environment_var("WANDB_ENTITY_ORG"), 'Environment variable "WANDB_ENTITY_ORG" is None or ""'
    assert validate_environment_var("WANDB_PROJECT"), 'Environment variable "WANDB_PROJECT" is None or ""'

    logger.info("All environment variables exist")


