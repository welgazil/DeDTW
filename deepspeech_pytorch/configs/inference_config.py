from dataclasses import dataclass

from deepspeech_pytorch.enums import DecoderType


@dataclass
class LMConfig:
    decoder_type: DecoderType = DecoderType.greedy
    lm_path: str = ""  # Path to an (optional) kenlm language model for use with beam search (req\'d with trie)
    top_paths: int = 1  # Number of beams to return
    alpha: float = 0.0  # Language model weight
    beta: float = 0.0  # Language model word bonus (all words)
    cutoff_top_n: int = 40  # Cutoff_top_n characters with highest probs in vocabulary will be used in beam search
    cutoff_prob: float = 1.0  # Cutoff probability in pruning,default 1.0, no pruning.
    beam_width: int = 10  # Beam width to use
    lm_workers: int = 4  # Number of LM processes to use


@dataclass
class ModelConfig:
    precision: int = 32  # Set to 16 to use mixed-precision for inference
    cuda: bool = True
    model_path: str = ""


@dataclass
class InferenceConfig:
    lm: LMConfig = LMConfig()
    model: ModelConfig = ModelConfig()


@dataclass
class TranscribeConfig(InferenceConfig):
    audio_path: str = ""  # Audio file to predict on
    offsets: bool = False  # Returns time offset information


@dataclass
class EvalConfig(InferenceConfig):
    test_path: str = ""  # Path to validation manifest csv or folder
    verbose: bool = True  # Print out decoded output and error of each sample
    save_output: str = ""  # Saves output of model from test to this file_path
    batch_size: int = 20  # Batch size for testing
    num_workers: int = 4


@dataclass
class AugmentationConfig:
    speed_volume_perturb: bool = False  # Use random tempo and gain perturbations.
    spec_augment: bool = False  # Use simple spectral augmentation on mel spectograms.
    noise_dir: str = (
        ""  # Directory to inject noise into audio. If default, noise Inject not added
    )
    noise_prob: float = 0.4  # Probability of noise being added per sample
    noise_min: float = 0.0  # Minimum noise level to sample from. (1.0 means all noise, not original signal)
    noise_max: float = 0.5  # Maximum noise levels to sample from. Maximum 1.0
    noise_levels: tuple = (0, 0.5)
    gaussian_noise: bool = True


@dataclass
class EvalDTWConfig(InferenceConfig):
    test_path: str = "/gpfswork/rech/jnf/urm17su/deepdtw/test_triplets_all.csv"  # Path to validation manifest csv or folder
    human_test_csv: str = "/gpfswork/rech/jnf/urm17su/deepdtw/data_triplets/all_human_experimental_data.csv"  # Path to human test #csv
    verbose: bool = True  # Print out decoded output and error of each sample
    train_dir: str = (
        "/gpfswork/rech/jnf/urm17su/deepdtw/data_triplets/Perceptimatic/wavs_extracted"
    )
    transform_wav: bool = False
    transform_spect: bool = False
    save_output: str = ""  # Saves output of model from test to this file_path
    batch_size: int = 1  # Batch size for testing
    num_workers: int = 4
    representation: str = "dtw"
    augmentation: AugmentationConfig = AugmentationConfig()


@dataclass
class ServerConfig(InferenceConfig):
    host: str = "0.0.0.0"
    port: int = 8888
