# `eso.ESO`

The top-level orchestrator. Pass the path of a JSON settings file. Call `run()`. The data preparation, baseline training, evolution loop, and evaluation stages are all driven from inside this class.

A typical invocation looks like:

```python
from eso import ESO

e = ESO(settings_path="settings/my_experiment.json")
e.run()
```

## What `run()` does

The `run()` method is a thin wrapper that walks the full pipeline.

| Step | Method called | Description |
| --- | --- | --- |
| 1 | `_load_settings` | Validate the JSON against `eso.utils.settings`. |
| 2 | `_setup_logging` | Configure file and TensorBoard loggers. |
| 3 | `_prepare_data` | Build the preprocessed and unprocessed mel-spectrogram datasets. |
| 4 | `_train_baseline` | Train the baseline CNN on the preprocessed dataset. Record its F1 and parameter count. |
| 5 | `_initialise_population` | Create a random population of chromosomes. |
| 6 | `optimize()` | Iterate `max_generations` of selection, mutation, crossover, and reproduction. |
| 7 | `evaluate()` | Run sliding-window inference on the test set with the best chromosome. |

The methods marked as private (prefixed with `_`) are filtered out of the public reference below, but the public surface (`run`, `optimize`, `evaluate`) is documented in full.

## Settings

`ESO(settings_path=...)` is the only required argument. The JSON file is parsed into the dataclass hierarchy in [`eso.utils.settings`](utils.md). See [Configuration](../configuration.md) for every field.

## Reference

::: eso.eso.ESO
