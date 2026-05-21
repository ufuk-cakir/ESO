# Utilities

Preprocessing, annotation parsing, settings schema, evaluation, logging, and CPU-safe unpickling.

| Symbol | File | Role |
| --- | --- | --- |
| `Preprocessing` | `eso/utils/preprocessing.py` | Audio loading, optional filtering, mel-spectrogram generation. |
| `AnnotationReader` | `eso/utils/AnnotationReader.py` | Parse SVL or compatible XML annotation files. |
| `Config` and friends | `eso/utils/settings.py` | Typed configuration schema. One dataclass per section of the JSON. |
| `Evaluation` | `eso/utils/Evaluation.py` | Sliding-window inference, bout reconstruction, comparison metrics. |
| `plot_chromosome` · `setup_logger` · `log_tensorboard` | `eso/utils/logger.py` | Visualisation and logging helpers. |
| `CPU_Unpickler` | `eso/utils/unpickler.py` | Unpickle GPU-trained tensors onto CPU. |

## `eso.utils.preprocessing`

The audio-to-spectrogram pipeline. The class produces two datasets per species: a preprocessed one (low-pass filtered and downsampled, used to train the baseline) and an unprocessed one (used by ESO). Audio is segmented into fixed-length windows with a one-second overlap. Each segment is converted to a mel-spectrogram with a Hann window and a configurable hop length. Class balancing through time shifting, blending, and additive noise is also handled here.

::: eso.utils.preprocessing
    options:
      show_root_heading: false

## `eso.utils.AnnotationReader`

Parses Sonic Visualiser SVL files and equivalent XML annotation formats into a DataFrame of `(filename, start_time, end_time, label)` rows. The output is consumed by `Preprocessing` to mark presence and absence segments for training.

::: eso.utils.AnnotationReader
    options:
      show_root_heading: false

## `eso.utils.settings`

The typed configuration schema. The JSON passed to `ESO(settings_path=...)` is validated against these dataclasses. Each top-level section of the file maps to one class. Unknown fields raise a `ValueError` at load time.

For a narrative walk-through of every field with recommended values from the paper, see [Configuration](../configuration.md).

::: eso.utils.settings
    options:
      show_root_heading: false

## `eso.utils.Evaluation`

Reproduces the evaluation protocol described in the paper. The class slides a window over each test audio file, applies the model (baseline or ESO chromosome) per window, groups consecutive positive predictions into calling bouts, and computes true positives, false positives, false negatives, and true negatives using a 25 percent overlap rule (10 percent for the Thyolo Alethe dataset). It also measures FLOPs via `fvcore`, RAM usage via `psutil`, and energy via `CodeCarbon`.

::: eso.utils.Evaluation
    options:
      show_root_heading: false

## `eso.utils.logger`

Visualisation and logging. `plot_chromosome` renders the selected bands on top of a representative spectrogram, in the style of Figure 4 in the paper. `setup_logger` configures Python's standard logging to write to both a file and the console. `setup_tensorboard` and `log_tensorboard` push generation-level fitness scalars and the best chromosome's band layout to TensorBoard.

::: eso.utils.logger
    options:
      show_root_heading: false

## `eso.utils.unpickler`

`CPU_Unpickler` is a `pickle.Unpickler` subclass that redirects GPU-tensor loads to CPU. Use it when loading a chromosome saved on a CUDA host onto a CPU-only machine for inspection or inference.

::: eso.utils.unpickler
    options:
      show_root_heading: false
