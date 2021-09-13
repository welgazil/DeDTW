import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import csv
import librosa
import numpy as np
import sox
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader
import torchaudio

from deepspeech_pytorch.configs.train_config import SpectConfig, AugmentationConfig
from deepspeech_pytorch.loader.spec_augment import spec_augment

torchaudio.set_audio_backend("sox_io")


def load_audio(path):
    sound, sample_rate = torchaudio.load(path)
    if sound.shape[0] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=0)  # multiple channels, average
    return sound.numpy()


def load_audio_librosa(path):
    sound, sample_rate = librosa.load(path)
    return sound, sample_rate


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self, path=None, sample_rate=16000, noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = sox.file_info.duration(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_start = 0
        noise_end = noise_start + noise_len
        #print(self.sample_rate)
        noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        #print("lendata", len(data))
        #print("lennoise", len(noise_dst))
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data


class SpectrogramParser(AudioParser):
    def __init__(
        self,
        audio_conf: SpectConfig,
        augmentation_conf: AugmentationConfig ,
        normalize: bool = False,
        
    ):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augmentation_conf(Optional): Config containing the augmentation parameters
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = normalize
        self.aug_conf = augmentation_conf
        self.gaussian_noise = augmentation_conf.gaussian_noise
        
        if augmentation_conf and augmentation_conf.noise_dir:
            self.noise_injector = NoiseInjection(
                path=augmentation_conf.noise_dir,
                sample_rate=self.sample_rate,
                noise_levels=augmentation_conf.noise_levels,
            )
        else:
            self.noise_injector = None

    def parse_audio(self, audio_path):
        if self.aug_conf and self.aug_conf.speed_volume_perturb:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)
        if self.noise_injector:
            add_noise = np.random.binomial(1, self.aug_conf.noise_prob)
            if add_noise:
                y = self.noise_injector.inject_noise(y)
        if self.gaussian_noise==True:
            print('In gaussian NOISEEEEE #######################')
            wn = np.random.randn(len(y))
            y = y + 0.1 * wn
            

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
        )
        spect, phase = librosa.magphase(D)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        if self.aug_conf.spec_augment:
            spect = spec_augment(spect)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(
        self,
        audio_conf: SpectConfig,
        aug_cfg: AugmentationConfig,
        input_path: str,
        labels: list,
        normalize: bool = False,
        
    ):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...
        You can also pass the directory of dataset.
        :param audio_conf: Config containing the sample rate, window and the window length/stride in seconds
        :param input_path: Path to input.
        :param labels: List containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augmentation_conf(Optional): Config containing the augmentation parameters
        """
        self.ids = self._parse_input(input_path)
        self.size = len(self.ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf=audio_conf, normalize=normalize, augmentation_conf=aug_cfg)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        return spect, transcript

    def _parse_input(self, input_path):
        ids = []
        if os.path.isdir(input_path):
            for wav_path in Path(input_path).rglob("*.wav"):
                transcript_path = (
                    str(wav_path).replace("/wav/", "/txt/").replace(".wav", ".txt")
                )
                ids.append((wav_path, transcript_path))
        else:
            # Assume it is a manifest file
            with open(input_path) as f:
                manifest = json.load(f)
            for sample in manifest["samples"]:
                wav_path = os.path.join(manifest["root_path"], sample["wav_path"])
                transcript_path = os.path.join(
                    manifest["root_path"], sample["transcript_path"]
                )
                ids.append((wav_path, transcript_path))
        return ids

    def parse_transcript(self, transcript_path):
        with open(transcript_path, "r", encoding="utf8") as transcript_file:
            transcript = transcript_file.read().replace("\n", "")
        transcript = list(
            filter(None, [self.labels_map.get(x) for x in list(transcript)])
        )
        return transcript

    def __len__(self):
        return self.size


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class DSRandomSampler(Sampler):
    """
    Implementation of a Random Sampler for sampling the dataset.
    Added to ensure we reset the start index when an epoch is finished.
    This is essential since we support saving/loading state during an epoch.
    """

    def __init__(self, dataset, batch_size=1):
        super().__init__(data_source=dataset)

        self.dataset = dataset
        self.start_index = 0
        self.epoch = 0
        self.batch_size = batch_size
        ids = list(range(len(self.dataset)))
        self.bins = [
            ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)
        ]

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.bins) - self.start_index, generator=g)
            .add(self.start_index)
            .tolist()
        )
        for x in indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)
            yield batch_ids

    def __len__(self):
        return len(self.bins) - self.start_index

    def set_epoch(self, epoch):
        self.epoch = epoch


class DSElasticDistributedSampler(DistributedSampler):
    """
    Overrides the ElasticDistributedSampler to ensure we reset the start index when an epoch is finished.
    This is essential since we support saving/loading state during an epoch.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=1):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        self.start_index = 0
        self.batch_size = batch_size
        ids = list(range(len(dataset)))
        self.bins = [
            ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)
        ]
        self.num_samples = int(
            math.ceil(float(len(self.bins) - self.start_index) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.bins) - self.start_index, generator=g)
            .add(self.start_index)
            .tolist()
        )

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        for x in indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)
            yield batch_ids

    def __len__(self):
        return self.num_samples


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = (
            'sox "{}" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1'.format(
                path, sample_rate, tar_filename, start_time, end_time
            )
        )
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = [
            "tempo",
            "{:.3f}".format(tempo),
            "gain",
            "{:.3f}".format(gain),
        ]
        sox_params = 'sox "{}" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1'.format(
            path, sample_rate, augmented_filename, " ".join(sox_augment_params)
        )
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(
    path, sample_rate=16000, tempo_range=(0.85, 1.15), gain_range=(-6, 8)
):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(
        path=path, sample_rate=sample_rate, tempo=tempo_value, gain=gain_value
    )
    return audio


class DTWData(Dataset, SpectrogramParser):
    def __init__(
        self,
        audio_conf: SpectConfig,
        augmentation_conf: AugmentationConfig,
        train_csv: str,
        human_csv: str,
        train_dir: str,
        language: str,
        level: str,
        normalize: bool = False,
        adding_noise: bool = False
        
    ):
        self.level = level
        self.language = 'EN' if language == 'english' else 'FR'
        self.ids_train_df = self._parse_input_train(train_csv)
        self.human_triplet, self.human_contrast = self._parse_input_human(human_csv)
        self.train_dir = train_dir
        self.adding_noise = adding_noise
        aug_conf = augmentation_conf
        if not self.adding_noise and aug_conf.gaussian_noise:
            print('NOT ADDING NOISE FIRST')
            aug_conf.gaussian_noise = False
        else:
            print('NOOW ADDING NOISE')

        super(DTWData, self).__init__(audio_conf=audio_conf, normalize=normalize, augmentation_conf=aug_conf)

    def __getitem__(self, index):
        sample = self.ids_train_df[index]


        TGT_path = os.path.join(self.train_dir, sample['TGT_item'])
        OTH_path = os.path.join(self.train_dir, sample['OTH_item'])
        X_path = os.path.join(self.train_dir, sample['X_item'])
        dataset_triplet = sample['dataset']
        id_triplets = sample['triplet_id']


        if self.level == 'triplet':
            value = self.human_triplet[id_triplets]
            labels_all = [x[0] for x in value if (x[1] == dataset_triplet and x[2] == self.language)] # we check right name of dataset and triplet and language
        elif self.level == 'contrast':
            # we get contrast and language n top of dataset name
            triplet_cont1, triplet_cont2 = self.human_triplet[id_triplets][0][3], self.human_triplet[id_triplets][0][4]
            #print(triplet_cont1, triplet_cont2)
            if triplet_cont2 in self.human_contrast:
                value = self.human_contrast[triplet_cont2]
            elif triplet_cont1 in self.human_contrast:
                value = self.human_contrast[triplet_cont1]
            else:
                print('Problem', triplet_cont1, triplet_cont2)
            #print(value)
            #value = self.human_contrast.get(triplet_cont1, self.human_contrast[triplet_cont2])
            labels_all = []
            for x in value:
                if x[1] == self.language:
                    labels_all += [float(x[0])]
        else:
            print('Error, level not implemented')

        #print(labels_all)
        # compute sfft
        TGT = self.parse_audio(TGT_path)
        OTH = self.parse_audio(OTH_path)
        X = self.parse_audio(X_path)

        # according to the dataset we take the average result of the humans
        # We transform the labels so they are between 0 and 1 (0 is chance level for human, 1 is perfect score)
        if dataset_triplet == "WorldVowels" or dataset_triplet == "zerospeech":
            labels_list = [float(x) / 3.0 for x in labels_all]
            # print(labels_list)
            labels = np.mean(labels_list)
            # print(labels)
        else:
            labels_list = [float(x) for x in labels_all]
            labels = np.mean(labels_list)
        return TGT, OTH, X, id_triplets, labels
    
    def _parse_input_train(self,input_path):
        
        ids_list = []
        
        with open(input_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ids_list.append(row)
                
        return ids_list
    
    def _parse_input_human(self,input_path):
        ids_2 = {}
        ids_3 = {}
        with open(input_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                contrast1 = row['phone_TGT'] + ';' + row['phone_OTH'] + ';' + row['dataset'] + ';' + row[
                    'language_TGT'] + ';' + row['language_OTH']
                contrast2 = row['phone_OTH'] + ';' + row['phone_TGT'] + ';' + row['dataset'] + ';' + row[
                    'language_OTH'] + ';' + row['language_TGT']
                if row['triplet_id'] in ids_2.keys():
                    ids_2[row['triplet_id']].append((row['user_ans'],row['dataset'], row['subject_language'],contrast1, contrast2))
                else :
                    ids_2[row['triplet_id']] = [(row['user_ans'],row['dataset'], row['subject_language'],contrast1, contrast2)]

        with open(input_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                contrast1 = row['phone_TGT'] + ';' + row['phone_OTH'] + ';' + row['dataset'] + ';' + row['language_TGT'] + ';' + row['language_OTH']
                contrast2 = row['phone_OTH'] + ';' + row['phone_TGT'] + ';' + row['dataset'] + ';' + row['language_OTH'] + ';' + row['language_TGT']
                if contrast1 in ids_3.keys():
                    ids_3[contrast1].append((row['user_ans'],row['subject_language']))
                elif contrast2 in ids_3.keys():
                    ids_3[contrast2].append((row['user_ans'], row['subject_language']))
                else :
                    ids_3[contrast1] = [(row['user_ans'], row['subject_language'])]
        return ids_2, ids_3


    def __len__(self):
        return len(self.ids_train_df)

    


def _collate_fn_dtw(batch):
    data = [(item[0], item[1], item[2]) for item in batch]
    id_triplets = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    return data, id_triplets, labels


class AudioDTWDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDTWDataLoader, self).__init__(*args, **kwargs)

    #  self.collate_fn = _collate_fn_dtw
