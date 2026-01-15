import hydra
from hydra.utils import instantiate


@hydra.main(config_path="../../configs/", config_name="train", version_base=None)
def main(cfg):
    model = instantiate(cfg.models)
    print(model)

if __name__ == "__main__":
    main()