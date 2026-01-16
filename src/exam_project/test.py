import hydra
from hydra.utils import instantiate


@hydra.main(config_path="configs", config_name="test", version_base=None)
def test(cfg):
    model = instantiate(cfg.models)
    print(model)

if __name__ == "__main__":
    test()