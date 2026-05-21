# Model and data

Modules under `eso.model` cover the CNN architecture, the dataset pipeline, and the training and evaluation wrappers used by both the baseline and the per-chromosome models. See [CNN training and fitness](../concepts/cnn-training.md) for the algorithmic context.

| Symbol | File | Role |
| --- | --- | --- |
| `BaseCNN` | `eso/model/cnn.py` | The shared CNN architecture (1 conv layer, max-pool, two FC layers). |
| `calc_back_conv` | `eso/model/cnn.py` | Backward computation of the minimum legal input size through a convolution. |
| `calc_back_pool` | `eso/model/cnn.py` | The pooling counterpart of `calc_back_conv`. |
| `Data` | `eso/model/data.py` | Audio to spectrogram pipeline, splits, and dataset caching. |
| `Model` | `eso/model/model.py` | Train, evaluate, save, and load wrapper around the CNN. |

## `eso.model.cnn`

The CNN architecture. The default is a simple stack: one `Conv2d → ReLU → MaxPool` block followed by a flatten and two fully connected layers ending in a 2-unit softmax. Sizing is parameterised through [`ArchitectureConfig`](../configuration.md#cnn_architecture).

The module also exposes the helpers `calc_back_conv` and `calc_back_pool`, which walk the convolution and pool stack backwards to compute the smallest input shape the network can accept. ESO uses these helpers at startup to validate every gene's height against the architecture, so impossible configurations fail fast rather than during training.

::: eso.model.cnn
    options:
      show_root_heading: false

## `eso.model.data`

The data pipeline. `Data.create_datasets` reads audio, parses annotations, segments to fixed-length windows, generates mel-spectrograms, applies optional class balancing via augmentation, and writes the train/validation/test splits to disk (or holds them in memory if `keep_in_memory` is set). The same splits are reused across the baseline and every chromosome's CNN training, so all individuals see identical data.

::: eso.model.data
    options:
      show_root_heading: false

## `eso.model.model`

The training and inference wrapper around a CNN. `Model` is used both by `ESO` to train the baseline and by every `Chromosome` to train its own CNN on the extracted bands. It owns the optimiser, the loss function, and the early-stopping logic. It also implements `get_number_of_parameters`, which produces the parameter count used in the fitness equation.

::: eso.model.model
    options:
      show_root_heading: false
