from exam_project.model import BaseCNN, BaseANN, ViTClassifier

from loguru import logger
import os
import wandb


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


def get_device_from_artifact(artifact):
    """Gets device from artifact through metadata"""
    return artifact.metadata.get("device")

def load_model_from_wandb(artifact: str, alias: str = 'production') -> BaseANN | BaseCNN | ViTClassifier:
    """
    Loads model from a wandb artifact

    Params:
        - artifact (str): artifact string.
            Example: "krusand-danmarks-tekniske-universitet-dtu-org/wandb-registry-fer-model/cnn:production"
        - alias (str): The wandb alias of the model
            Examples: 'production', 'staging', 'v1', 'latest'

    Returns:
        One of BaseANN, BaseCNN or ViTClassifier, loaded with artifact checkpoint    
    """

    MODELS = {
        'ann': BaseANN,
        'cnn': BaseCNN,
        'vit': ViTClassifier
    }
    logger.info("Loading model artifact from WandB")
    logger.info("Function inputs:")
    logger.info(f"{artifact = }, {alias = }")
    
    model_architecture = os.getenv("MODEL_ARCHITECTURE")
    model = MODELS.get(model_architecture)

    logger.info(f"{model = }")    
    assert model_architecture in list(MODELS.keys()), f"Model architecture '{model_architecture}' not supported."

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY_ORG")
                   , "project": os.getenv("WANDB_PROJECT")},
    )

    logger.info("API connection established")
    
    artifact_name_version = f"{os.getenv("MODEL_NAME")}"
    logger.info(f"{artifact_name_version = }")

    artifact_name, artifact_version = artifact_name_version.split(":")
    artifact = api.artifact(f"{artifact_name}:{alias}", type="Model")

    logger.info("Downloading artifact")
    artifact.download(root="./artifacts")
    logger.info("Artifact downloaded")

    file_name = artifact.files()[0].name

    logger.info(f"{file_name = }")

    return model.load_from_checkpoint(f"./artifacts/{file_name}"), artifact
