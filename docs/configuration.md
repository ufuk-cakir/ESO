# Configuration

ESO is driven by a single JSON file. The path must end in `.json`. The schema is enforced by typed dataclasses in [`eso.utils.settings`](api/utils.md). Unknown fields raise a `ValueError` at load time.

## Entry point

```python
from eso import ESO
ESO(settings_path="settings/my_experiment.json").run()
```

Editable templates live in `settings/` at the repository root. The default values shown below come from [`eso.utils.settings`](api/utils.md). The recommended values, where they differ, are the values used in the published experiments.

## File structure

```json
{
  "algorithm":          { ... },
  "population":         { ... },
  "selection_operator": { ... },
  "genetic_operator":   { ... },
  "gene":               { ... },
  "chromosome":         { ... },
  "model":              { ... },
  "cnn_architecture":   { ... },
  "data":               { ... },
  "preprocessing":      { ... }
}
```

Each section maps to one dataclass. Missing fields take their defaults.

## algorithm

Top-level GA control.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_generations` | `int` | `100` | Number of generations to evolve the population for. The paper uses `20`. |

```json
"algorithm": { "max_generations": 20 }
```

## population

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `pop_size` | `int` | `10` | Number of chromosomes per generation. Held constant across generations. The paper uses `300`. |

```json
"population": { "pop_size": 300 }
```

## selection_operator

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `tournament_size` | `int` | `10` | Number of chromosomes sampled for each tournament. The best of the sample becomes a parent. |

```json
"selection_operator": { "tournament_size": 5 }
```

See [Parent selection](concepts/evolution.md#parent-selection).

## genetic_operator

Operator rates and mutation step sizes. See [Operators](concepts/evolution.md#operators).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `mutation_rate` | `float` | `0.1` | Probability of generating an offspring through mutation. |
| `crossover_rate` | `float` | `0.8` | Probability of generating offspring through crossover. |
| `reproduction_rate` | `float` | `0.1` | Probability of copying a parent unchanged into the next generation. |
| `mutation_height_range` | `int` | `5` | Maximum $|\delta h|$ added to a gene's height during a height mutation. |
| `mutation_position_range` | `int` | `20` | Maximum $|\delta P|$ added to a gene's position during a position mutation. |

The three rates must sum to 1. The paper's reported configuration is `0.3 / 0.6 / 0.1` for mutation, crossover, reproduction.

```json
"genetic_operator": {
  "mutation_rate": 0.3,
  "crossover_rate": 0.6,
  "reproduction_rate": 0.1,
  "mutation_height_range": 5,
  "mutation_position_range": 20
}
```

## gene

Constraints on the position $P_k$ and height $h_k$ of every gene. See [Gene](concepts/genes-and-chromosomes.md#gene).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `min_position` | `int` | `0` | Lower bound on $P_k$. |
| `max_position` | `int` | `-1` | Upper bound on $P_k$. `-1` defaults to the spectrogram height $S_h$. |
| `min_height` | `int` | `4` | Lower bound on $h_k$. |
| `max_height` | `int` | `16` | Upper bound on $h_k$. |
| `band_position` | `int \| null` | `null` | Fix $P_k$ to a single value for every gene. Use `null` or `-1` to disable. |
| `band_height` | `int \| null` | `null` | Fix $h_k$ to a single value for every gene. Use `null` or `-1` to disable. |
| `spec_height` | `int \| null` | `null` | Spectrogram height $S_h$. Filled in automatically from preprocessing. |
| `minimum_gene_height` | `int \| null` | `null` | Minimum legal height given the convolution stack. Computed automatically. |

```json
"gene": {
  "min_position": 0,
  "max_position": -1,
  "min_height": 1,
  "max_height": 16
}
```

## chromosome

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_genes` | `int \| null` | `null` | Fix the number of genes per chromosome. Use `null` or `-1` to draw from `[min_num_genes, max_num_genes]`. |
| `min_num_genes` | `int` | `3` | Lower bound on the number of genes when `num_genes` is disabled. |
| `max_num_genes` | `int` | `10` | Upper bound on the number of genes when `num_genes` is disabled. |
| `lambda_1` | `float` | `0.5` | Weight of the F1 term in the fitness function. |
| `lambda_2` | `float` | `0.5` | Weight of the parameter term in the fitness function. |
| `stack` | `bool` | `false` | If `true`, extracted bands are stacked along a depth axis and all heights must be equal. If `false`, bands are concatenated along the frequency axis. |
| `baseline_parameters` | `float \| null` | `null` | Filled in automatically from the trained baseline. |
| `baseline_metric` | `int \| null` | `null` | Filled in automatically from the trained baseline. |

See [Fitness](concepts/cnn-training.md#fitness) for the equation. Paper values: $\lambda_1 = 0.95, \lambda_2 = 0.05$, except for Hainan gibbon ($\lambda_1 = 0.99, \lambda_2 = 0.01$).

```json
"chromosome": {
  "min_num_genes": 1,
  "max_num_genes": 10,
  "lambda_1": 0.95,
  "lambda_2": 0.05,
  "stack": false
}
```

## model

Training hyperparameters used by both the baseline and per-chromosome CNNs.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer_name` | `str` | `"adam"` | Optimiser. Currently `adam`. |
| `loss_function_name` | `str` | `"cross_entropy"` | Loss function. |
| `num_epochs` | `int` | `1` | Training epochs per CNN. The paper uses `30`. |
| `batch_size` | `int` | `128` | Mini-batch size. The paper uses `64`. |
| `learning_rate` | `float` | `0.001` | Learning rate for the optimiser. |
| `shuffle` | `bool` | `true` | Whether to shuffle batches during training. |
| `metric` | `str` | `"f1"` | Validation metric used as the F1 term in fitness. Supported values: `f1`, `accuracy`. |

```json
"model": {
  "optimizer_name": "adam",
  "loss_function_name": "cross_entropy",
  "num_epochs": 30,
  "batch_size": 64,
  "learning_rate": 0.001,
  "metric": "f1"
}
```

## cnn_architecture

CNN topology shared by baseline and per-chromosome models.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `conv_layers` | `int` | `1` | Number of `Conv2d` blocks. |
| `conv_filters` | `int` | `8` | Filters per convolutional layer. |
| `conv_kernel` | `int` | `8` | Kernel size of each convolutional filter. |
| `conv_padding` | `str \| null` | `null` | Padding strategy. `null` defaults to no padding. |
| `max_pooling_size` | `int` | `4` | Window size of `MaxPool2d`. |
| `stride_maxpool` | `int \| null` | `null` | Stride for `MaxPool2d`. `null` matches `max_pooling_size`. |
| `fc_layers` | `int` | `2` | Number of fully-connected layers before the output. |
| `fc_units` | `int` | `32` | Units per fully-connected layer. |
| `dropout_rate` | `float` | `0.5` | Dropout applied after each fully-connected layer. |

The paper's baseline corresponds to `conv_layers = 1`, `conv_filters = 8`, `conv_kernel = 8`, `max_pooling_size = 4`, `fc_layers = 2`, `fc_units = 32`.

```json
"cnn_architecture": {
  "conv_layers": 1,
  "conv_filters": 8,
  "conv_kernel": 8,
  "max_pooling_size": 4,
  "fc_layers": 2,
  "fc_units": 32,
  "dropout_rate": 0.5
}
```

## data

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `species_folder` | `str` | `""` | Absolute path to the species' dataset directory. |
| `positive_class` | `str` | `""` | Folder or label of the positive class. |
| `negative_class` | `str` | `""` | Folder or label of the negative class. |
| `train_size` | `float` | `0.8` | Fraction of files in the training split. |
| `test_size` | `float` | `0.2` | Fraction of files in the test split. Validation gets the remainder. |
| `reshuffle` | `bool` | `false` | If `true`, re-randomises file assignment on every run. |
| `keep_in_memory` | `bool` | `false` | If `true`, holds spectrograms in RAM. Faster but memory-bound. |
| `force_recreate_dataset` | `bool` | `false` | If `true`, regenerates the cached dataset from audio. |

```json
"data": {
  "species_folder": "/data/gibbons",
  "positive_class": "gibbon",
  "negative_class": "no-gibbon",
  "train_size": 0.6,
  "test_size": 0.2
}
```

## preprocessing

Mel-spectrogram generation. See [Spectrogram preprocessing](concepts/preprocessing.md).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `sample_rate` | `int` | `32000` | Original recording sample rate. |
| `lowpass_cutoff` | `int` | `2000` | Cut-off frequency of the low-pass filter applied to the baseline dataset. |
| `downsample_rate` | `int` | `4800` | Downsample target for the baseline dataset. Set to twice the Nyquist rate. |
| `nyquist_rate` | `int` | `2400` | Maximum frequency in the target species' calls. |
| `segment_duration` | `int` | `4` | Fixed window length in seconds for segmentation. |
| `nb_negative_class` | `int` | `20` | Number of negative segments to extract per audio file. |
| `file_type` | `str` | `"svl"` | Annotation format. `svl` or compatible XML. |
| `audio_extension` | `str` | `".wav"` | File extension of audio recordings. |
| `n_fft` | `int` | `1024` | Hann window size in samples for the STFT. |
| `hop_length` | `int` | `256` | Stride between consecutive STFT frames in samples. |
| `n_mels` | `int` | `128` | Number of mel bands. Sets the spectrogram height $S_h$. |
| `f_min` | `int` | `4000` | Minimum frequency for the mel filter bank. |
| `f_max` | `int` | `9000` | Maximum frequency for the mel filter bank. |

Per-species values from the paper are listed in [Spectrogram preprocessing](concepts/preprocessing.md#spectrograms).

```json
"preprocessing": {
  "sample_rate": 9600,
  "lowpass_cutoff": 2000,
  "downsample_rate": 4800,
  "nyquist_rate": 2400,
  "segment_duration": 4,
  "n_fft": 1024,
  "hop_length": 256,
  "n_mels": 128,
  "f_min": 0,
  "f_max": 2000,
  "file_type": "svl",
  "audio_extension": ".wav"
}
```

## Full example: Hainan gibbon

Below is a runnable configuration that mirrors the published experiment for the Hainan gibbon dataset.

```json
{
  "algorithm": { "max_generations": 20 },
  "population": { "pop_size": 300 },
  "selection_operator": { "tournament_size": 5 },
  "genetic_operator": {
    "mutation_rate": 0.3,
    "crossover_rate": 0.6,
    "reproduction_rate": 0.1
  },
  "gene": {
    "min_position": 0,
    "max_position": -1,
    "min_height": 1,
    "max_height": 16
  },
  "chromosome": {
    "min_num_genes": 1,
    "max_num_genes": 10,
    "lambda_1": 0.99,
    "lambda_2": 0.01,
    "stack": false
  },
  "model": {
    "optimizer_name": "adam",
    "loss_function_name": "cross_entropy",
    "num_epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
    "metric": "f1"
  },
  "cnn_architecture": {
    "conv_layers": 1,
    "conv_filters": 8,
    "conv_kernel": 8,
    "max_pooling_size": 4,
    "fc_layers": 2,
    "fc_units": 32,
    "dropout_rate": 0.5
  },
  "data": {
    "species_folder": "/data/Hainan_gibbon",
    "positive_class": "gibbon",
    "negative_class": "no-gibbon",
    "train_size": 0.6,
    "test_size": 0.2
  },
  "preprocessing": {
    "sample_rate": 9600,
    "lowpass_cutoff": 2000,
    "downsample_rate": 4800,
    "nyquist_rate": 2400,
    "segment_duration": 4,
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 128,
    "f_min": 0,
    "f_max": 5000,
    "file_type": "svl",
    "audio_extension": ".wav"
  }
}
```

## Auto-generated reference

For the raw dataclass definitions, see [`eso.utils.settings`](api/utils.md).
