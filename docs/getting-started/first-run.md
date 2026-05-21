# First run

A minimal experiment from the Python API. For a GUI walkthrough, see [GUI](gui.md).

## Prerequisites

- A directory of `.wav` files for the target species, organised by folder.
- Annotation files in SVL format alongside the audio. The parser is [`AnnotationReader`](../api/utils.md).
- A settings JSON file. A starting template is at `settings/settings.json` in the repository.

## Settings format

ESO reads JSON. The path must end with `.json`. The top-level keys map to dataclasses in [`eso.utils.settings`](../api/utils.md).

```json
{
  "algorithm": {
    "max_generations": 20
  },
  "population": {
    "pop_size": 300
  },
  "selection_operator": {
    "tournament_size": 5
  },
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
    "lambda_1": 0.95,
    "lambda_2": 0.05,
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
    "dropout_rate": 0.5,
    "fc_layers": 2,
    "fc_units": 32
  },
  "data": {
    "species_folder": "/absolute/path/to/your/dataset",
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
    "f_max": 2000,
    "file_type": "svl",
    "audio_extension": ".wav"
  }
}
```

For the full field reference, see [Configuration](../configuration.md).

## Run the pipeline

```python
from eso import ESO

eso = ESO(settings_path="settings/my_experiment.json")
eso.run()
```

`run()` performs the full pipeline.

1. Load and validate the settings JSON.
2. Build the two mel-spectrogram datasets. Preprocessed for the baseline. Unprocessed for ESO.
3. Train the baseline CNN on the preprocessed dataset. Record its F1 and parameter count.
4. Initialise a random population of chromosomes.
5. For each generation, train every chromosome's CNN, compute fitness, and produce the next population through reproduction, mutation, and crossover.
6. Track the best chromosome across all generations.
7. Evaluate the best chromosome on the test set with sliding-window inference. Write metrics and TensorBoard logs.

## Monitor in TensorBoard

```bash
tensorboard --logdir runs/
```

Open `http://localhost:6006`.

## Outputs

Each run writes a timestamped directory.

| Artifact | Contents |
| --- | --- |
| `best_chromosome.pkl` | Pickled best chromosome with genes and trained CNN weights. |
| `population_genN.pkl` | Population snapshots, one per generation. |
| `evaluation/` | Confusion matrix, F1, FLOPs, RAM, energy. |
| `tensorboard/` | TensorBoard event files. |

## Load a saved chromosome

```python
import pickle

with open("runs/2026-05-20_…/best_chromosome.pkl", "rb") as f:
    chromosome = pickle.load(f)

for gene in chromosome.get_genes():
    print(gene.get_band_position(), gene.get_band_height())

predictions = chromosome.predict(my_spectrogram_batch)
```

Models trained on GPU are unpickled to CPU automatically via [`eso.utils.unpickler.CPU_Unpickler`](../api/utils.md). A GPU is not required to inspect a finished run.

## Next

- [How ESO works](../concepts/overview.md). The reasoning behind each step.
- [Configuration](../configuration.md). Every field of the settings JSON.
