import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import EvalDTWConfig
from deepspeech_pytorch.testing import evaluate


cs = ConfigStore.instance()
cs.store(name="config", node=EvalDTWConfig)


@hydra.main(config_name="config")
def hydra_main(cfg: EvalDTWConfig):
    evaluate(cfg=cfg)


if __name__ == "__main__":
    hydra_main()
