import pytorch_lightning as pl
from hydra.utils import to_absolute_path
import numpy as np

from deepspeech_pytorch.configs.train_config import DTWDataConfig, AugmentationConfig
from deepspeech_pytorch.loader.data_loader import (
    SpectrogramDataset,
    DSRandomSampler,
    AudioDataLoader,
    DSElasticDistributedSampler,
)
from deepspeech_pytorch.loader.data_loader import DTWData, AudioDTWDataLoader
from torch.utils.data import ConcatDataset


# class DeepSpeechDataModule(pl.LightningDataModule):

#     def __init__(self,
#                  labels: list,
#                  data_cfg: DataConfig,
#                  normalize: bool,
#                  is_distributed: bool):
#         super().__init__()
#         self.train_path = to_absolute_path(data_cfg.train_path)
#         self.val_path = to_absolute_path(data_cfg.val_path)
#         self.labels = labels
#         self.data_cfg = data_cfg
#         self.spect_cfg = data_cfg.spect
#         self.aug_cfg = data_cfg.augmentation
#         self.normalize = normalize
#         self.is_distributed = is_distributed

#     def train_dataloader(self):
#         train_dataset = self._create_dataset(self.train_path)
#         if self.is_distributed:
#             train_sampler = DSElasticDistributedSampler(
#                 dataset=train_dataset,
#                 batch_size=self.data_cfg.batch_size
#             )
#         else:
#             train_sampler = DSRandomSampler(
#                 dataset=train_dataset,
#                 batch_size=self.data_cfg.batch_size
#             )
#         train_loader = AudioDataLoader(
#             dataset=train_dataset,
#             num_workers=self.data_cfg.num_workers,
#             batch_sampler=train_sampler
#         )
#         return train_loader

#     def val_dataloader(self):
#         val_dataset = self._create_dataset(self.val_path)
#         val_loader = AudioDataLoader(
#             dataset=val_dataset,
#             num_workers=self.data_cfg.num_workers,
#             batch_size=self.data_cfg.batch_size
#         )
#         return val_loader

#     def _create_dataset(self, input_path):
#         dataset = SpectrogramDataset(
#             audio_conf=self.spect_cfg,
#             input_path=input_path,
#             labels=self.labels,
#             normalize=True,
#             aug_cfg=self.aug_cfg
#         )

#         return dataset


class DeepSpeechDataModule(pl.LightningDataModule):
    def __init__(self, dataD_cfg: DTWDataConfig, augmentation : AugmentationConfig, is_distributed: bool):
        super().__init__()
        self.train_csv = to_absolute_path(dataD_cfg.train_csv)
        self.human_train_csv = to_absolute_path(dataD_cfg.human_train_csv)
        self.val_csv = to_absolute_path(dataD_cfg.val_csv)
        self.human_val_csv = to_absolute_path(dataD_cfg.human_val_csv)
        self.train_dir = to_absolute_path(dataD_cfg.train_dir)
        self.data_cfg = dataD_cfg
        self.spect_cfg = dataD_cfg.spect
        self.aug_cfg = augmentation
        self.is_distributed = is_distributed
        self.gaussian_noise = dataD_cfg.gaussian_noise
        
      
        print( 'lolol' ,self.aug_cfg)

    # self.is_distributed = None

    def train_dataloader(self):
        train_dataset = self._create_dataset(
            self.train_csv, self.human_train_csv, self.train_dir , self.gaussian_noise
        )
        
        print(self.gaussian_noise)
        if self.is_distributed:
            train_sampler = DSElasticDistributedSampler(
                dataset=train_dataset, batch_size=self.data_cfg.batch_size
            )
        else:
            train_sampler = DSRandomSampler(
                dataset=train_dataset, batch_size=self.data_cfg.batch_size
            )
        train_loader = AudioDTWDataLoader(
            dataset=train_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_sampler=train_sampler,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = self._create_dataset(
            self.val_csv, self.human_val_csv, self.train_dir, False
        )
        val_loader = AudioDTWDataLoader(
            dataset=val_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=self.data_cfg.batch_size,
        )
        return val_loader

    def _create_dataset(self, train_csv, human_csv, train_dir,gaussian_noise):
        # np.random.seed(42)
        dataset = DTWData(
            audio_conf=self.spect_cfg,
            train_csv=train_csv,
            human_csv=human_csv,
            train_dir=train_dir,
            augmentation_conf=self.aug_cfg,
            gaussian_noise=True
        )

        # data_augmentation
        
        

        if gaussian_noise:
            
            
            dataset_gaussian_noise = DTWData(
                audio_conf=self.spect_cfg,
                train_csv=train_csv,
                human_csv=human_csv,
                train_dir=train_dir,
                augmentation_conf=self.aug_cfg,
                gaussian_noise=False
            )
            dataset = ConcatDataset([dataset_gaussian_noise, dataset])

        return dataset
