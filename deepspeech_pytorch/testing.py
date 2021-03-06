import hydra
import torch

from deepspeech_pytorch.configs.inference_config import EvalDTWConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.configs.train_config import DTWDataConfig
from deepspeech_pytorch.loader.data_loader import (
    SpectrogramDataset,
    AudioDTWDataLoader,
    DTWData,
)
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.validation import run_evaluation, run_evaluationdtw


@torch.no_grad()
def evaluate(cfg: EvalDTWConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(device=device, model_path=cfg.model.model_path)

    test_dataset = DTWData(
        audio_conf=model.spect_cfg,
        train_csv=hydra.utils.to_absolute_path(cfg.test_path),
        human_csv=hydra.utils.to_absolute_path(cfg.human_test_csv),
        train_dir=hydra.utils.to_absolute_path(cfg.train_dir),
        augmentation_conf=cfg.augmentation,
        language=cfg.language_participants,
        level=cfg.level,
        adding_noise=False
    )

    # if cfg.augmentation and cfg.augmentation.gaussian_noise:

    #        dataset_gaussian_noise = DTWData(
    #               audio_conf=model.spect_cfg,
    #              train_csv=hydra.utils.to_absolute_path(cfg.test_path),
    #             human_csv=hydra.utils.to_absolute_path(cfg.human_test_csv),
    #            train_dir=hydra.utils.to_absolute_path(cfg.train_dir),
    #           aug_cfg = cfg.augmentation,

    #          )
    #  test_dataset = ConcatDataset([test_dataset,dataset_gaussian_noise])

    test_loader = AudioDTWDataLoader(
        test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    spearman, pearson, delta_positivity = run_evaluationdtw(
        human_csv=cfg.human_test_csv,
        test_loader=test_loader,
        device=device,
        model=model,
        precision=cfg.model.precision,
        representation=cfg.representation,
    )

    print(
        "Test Summary \t"
        "Spearman {spearman:.3f}\t"
        "Pearson  {pearson:.3f}\t"
        "Delta positivity  {delta_positivity:.3f}\t".format(
            spearman=spearman, pearson=pearson, delta_positivity=delta_positivity
        )
    )
