import wandb

api = wandb.Api()
artifact_path = "krusand-danmarks-tekniske-universitet-dtu/MLOps-exam/model-ezd38e3n:v0"
artifact = api.artifact(artifact_path)
artifact.link(target_path="krusand-danmarks-tekniske-universitet-dtu-org/wandb-registry-fer-model/Model new")
artifact.save()

# run = wandb.init(entity="krusand-danmarks-tekniske-universitet-dtu", project="MLOps-exam")
# artifact = run.use_artifact('krusand-danmarks-tekniske-universitet-dtu/MLOps-exam/model-brqq2gd7:v0', type='model')
# artifact_dir = artifact.download()
# run.link_artifact(
#     artifact=artifact,
#     target_path="model-registry/Fer-model",
#     aliases=["latest"]
# )