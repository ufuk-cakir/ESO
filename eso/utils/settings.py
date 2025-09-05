import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class BaseConfig:
    def dict(self):
        return asdict(self)


@dataclass
class AlgorithmConfig(BaseConfig):
    max_generations: int = 100


@dataclass
class GeneticOperatorConfig(BaseConfig):
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    reproduction_rate: float = 0.1
    mutation_height_range: int = 5
    mutation_position_range: int = 20


@dataclass
class SelectionOperatorConfig(BaseConfig):
    tournament_size: int = 10


@dataclass
class DataConfig(BaseConfig):
    force_recreate_dataset: bool = False
    keep_in_memory: bool = False
    species_folder: str = ""
    train_size: float = 0.8
    test_size: float = 0.2
    reshuffle: bool = False
    positive_class: str = ""
    negative_class: str = ""


@dataclass
class PreprocessingConfig(BaseConfig):
    sample_rate: int = 32000
    lowpass_cutoff: int = 2000
    downsample_rate: int = 4800
    nyquist_rate: int = 2400
    segment_duration: int = 4
    nb_negative_class: int = 20
    file_type: str = "svl"
    audio_extension: str = ".wav"
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: int = 4000
    f_max: int = 9000
    f_min_chromosome: int = 0
    f_max_chromosome: int = 5000


@dataclass
class PopulationConfig(BaseConfig):
    pop_size: int = 10


@dataclass
class GeneConfig(BaseConfig):
    min_position: int = 0
    max_position: int = -1
    min_height: int = 4
    max_height: int = 16
    band_position: int = None
    band_height: int = None
    spec_height: int = None
    minimum_gene_height: int = None


@dataclass
class ChromosomeConfig(BaseConfig):
    num_genes: int = None
    min_num_genes: int = 3
    max_num_genes: int = 10
    lambda_1: float = 0.5
    lambda_2: float = 0.5
    stack: bool = False
    baseline_parameters: float = None
    baseline_metric: int = None


@dataclass
class ModelConfig(BaseConfig):
    optimizer_name: str = "adam"
    loss_function_name: str = "cross_entropy"
    num_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.001
    shuffle: bool = True
    metric: str = "f1"

@dataclass
class ArchitectureConfig(BaseConfig):
    conv_layers: int = 1
    conv_filters: int = 8
    dropout_rate: float = 0.5
    conv_kernel: int = 8
    max_pooling_size: int = 4
    fc_units: int = 32
    fc_layers: int = 2
    conv_padding: str = None
    stride_maxpool: int = None


@dataclass
class Config(BaseConfig):
    _input: str = field(default=None)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    genetic_operator: GeneticOperatorConfig = field(
        default_factory=GeneticOperatorConfig
    )
    selection_operator: SelectionOperatorConfig = field(
        default_factory=SelectionOperatorConfig
    )
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    gene: GeneConfig = field(default_factory=GeneConfig)
    chromosome: ChromosomeConfig = field(default_factory=ChromosomeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cnn_architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)

    def __post_init__(self):
        if self._input is None:
            # Use default settings
            return
        # Check if input is path to a json
        if isinstance(self._input, str):
            self._load_from_json_path(self._input)
        if isinstance(self._input, dict):
            self._set_settings(self._input)

    def get_params(self):
        params = {}
        for key, value in asdict(self).items():
            if key == "_input":
                # params["settings"] = value
                continue
            for sub_key, sub_value in value.items():
                params[f"{key}_{sub_key}"] = sub_value
        return params

    def _load_from_json_path(self, path):
        if path.endswith(".json"):
            json_path = path
        else:
            raise ValueError("Can only load from JSON path.")
        if not os.path.exists(json_path):
            raise ValueError(f"Settings File not Found at {json_path}.")
        # Load File
        with open(json_path, "r") as f:
            data = json.load(f)
        self._set_settings(data)

    def _set_settings(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                config_class = getattr(self, key)
                for sub_key, sub_value in value.items():
                    if hasattr(config_class, sub_key):
                        setattr(config_class, sub_key, sub_value)
                    else:
                        raise ValueError(
                            f"The {sub_key} setting you are trying to set in {key} is not valid."
                        )
