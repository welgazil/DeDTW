import hydra
import torch

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.validation import run_evaluation,run_evaluationdtw


@torch.no_grad()
def evaluate(cfg: EvalConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

   # decoder = load_decoder(
    #    labels=model.labels,
     #   cfg=cfg.lm
   # )
   # target_decoder = GreedyDecoder(
    #    labels=model.labels,
     #   blank_index=model.labels.index('_')
   # )
    test_dataset = SpectrogramDataset(
        audio_conf=model.spect_cfg,
        train_csv=hydra.utils.to_absolute_path(cfg.test_path),
        human_csv=hydra.utils.to_absolute_path(cfg.human_test_csv),
    )
    test_loader = AudioDataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    
    spearman, pearson, delta_positivity = run_evaluationdtw(
        test_loader=test_loader,
        device=device,
        model=model,
        precision=cfg.model.precision
        
        )
    

    print('Test Summary \t'
          'Spearman {spearman:.3f}\t'
          'Pearson  {pearson:.3f}\t'
          'Delta positivity  {delta_positivity:.3f}\t'
          .format(spearman=spearman, pearson=pearson,delta_positivity=delta_positivity))
