# API reference

Generated from docstrings in the `eso` package. The conceptual explanation lives in [How ESO works](../concepts/overview.md). This section is the precise interface.

The reference is organised by subpackage. Each page lists every public class and function, the signature, parameters, return types, and raised exceptions. Private names (prefixed with `_`) are omitted.

| Page | Contents |
| --- | --- |
| [`eso.ESO`](eso.md) | The orchestrator that drives data preparation, baseline training, evolution, and evaluation. |
| [Genetic algorithm](ga.md) | The GA itself: `Gene`, `Chromosome`, `Population`, `SelectionOperator`, `GeneticOperator`. |
| [Model and data](model.md) | CNN architecture, data pipeline, training and evaluation wrappers. |
| [Utilities](utils.md) | Preprocessing, annotation parsing, settings schema, evaluation, logging, unpickling. |

## Conventions

- **Object kind chips.** Each entry is tagged `class`, `method`, `func`, `attr`, `module`, or `var` to identify its kind at a glance.
- **Signatures** are formatted with Black and shown on multiple lines for readability.
- **Parameter tables** list every documented argument with type, description, and default value (`required` if no default).
- **Cross-references.** Type annotations link to the corresponding object where one is available.
- **Source links.** Each entry exposes a "show source" disclosure with the relevant lines from the implementation.
