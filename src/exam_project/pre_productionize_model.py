import typer
import os
import wandb

from dotenv import load_dotenv
load_dotenv()

def productionize_artifact(artifact_path: str) -> None:
    """
    Promotes model to production alias

    Args:
        artifact_path: Path to the artifact to stage.
            Should be of the format "entity/project/artifact_name:version".
    """
    
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY_ORG")
                   , "project": os.getenv("WANDB_PROJECT")},
    )

    wandb_entity, wandb_registry, artifact_name_version = artifact_path.split("/")

    artifact_name, artifact_version = artifact_name_version.split(":")
    artifact = api.artifact(artifact_path)
    current_aliases = set(artifact.aliases)

    if 'production' in current_aliases:
        typer.echo(f"Info: {artifact.name} is already tagged with 'production'")
    else:
        current_aliases.add('production')
        typer.echo(f"Promoting {artifact.name} to 'production'")

    if 'staging' in current_aliases:
        current_aliases.remove('staging')
        typer.echo(f"Removing 'staging' tag from {artifact.name}")

    artifact.aliases=list(current_aliases)
    artifact.save()

if __name__ == "__main__":
    typer.run(productionize_artifact)