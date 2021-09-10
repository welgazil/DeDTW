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
from torch.utils.data import ConcatDataset,DataLoader


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
    def __init__(self, dataD_cfg: DTWDataConfig, is_distributed: bool):
        super().__init__()
        self.train_csv = to_absolute_path(dataD_cfg.train_csv)
        self.human_csv = to_absolute_path(dataD_cfg.human_csv)
        self.val_csv = to_absolute_path(dataD_cfg.val_csv)
        self.train_dir = to_absolute_path(dataD_cfg.train_dir)
        self.data_cfg = dataD_cfg
        self.spect_cfg = dataD_cfg.spect
        self.aug_cfg = dataD_cfg.aug
        self.is_distributed = is_distributed
        
      
        print( 'lolol' ,self.aug_cfg)

    def train_dataloader(self):
        train_dataset = self._create_dataset(
            self.train_csv, self.human_csv, self.train_dir
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
        """train_loader = AudioDTWDataLoader(
            dataset=train_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_sampler=train_sampler,
        )"""
        
        train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = self._create_dataset(
            train_csv=self.val_csv, human_csv=self.human_csv, train_dir=self.train_dir
        )
        """val_loader = AudioDTWDataLoader(
            dataset=val_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=self.data_cfg.batch_size,
        )"""
        
        val_loader = DataLoader(val_dataset,batch_size=1,shuffle=True)
        return val_loader

    def _create_dataset(self, train_csv, human_csv, train_dir):
        # np.random.seed(42)
        dataset = DTWData(
            audio_conf=self.spect_cfg,
            train_csv=train_csv,
            human_csv=human_csv,
            train_dir=train_dir,
            augmentation_conf=self.aug_cfg,
            language = self.data_cfg.language_participants,
            level=self.data_cfg.level

        )

        # data_augmentation
        if self.aug_cfg.gaussian_noise:
            dataset_gaussian_noise = DTWData(
                audio_conf=self.spect_cfg,
                train_csv=train_csv,
                human_csv=human_csv,
                train_dir=train_dir,
                augmentation_conf=self.aug_cfg,
                language=self.data_cfg.language_participants,
                level=self.data_cfg.level
            )
            dataset = ConcatDataset([dataset_gaussian_noise, dataset])

        return dataset
