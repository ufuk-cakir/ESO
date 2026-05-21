# Genetic algorithm

Modules under `eso.ga` implement the genetic algorithm. The representation is split across `Gene`, `Chromosome`, and `Population`. The variation machinery lives in `SelectionOperator` and `GeneticOperator`. For the algorithmic explanation, see [Genes and chromosomes](../concepts/genes-and-chromosomes.md) and [Evolution](../concepts/evolution.md).

| Symbol | File | Role |
| --- | --- | --- |
| `Gene` | `eso/ga/gene.py` | One horizontal band, encoded as `(band_position, band_height)`. |
| `Chromosome` | `eso/ga/chromosome.py` | Set of genes plus a trained CNN. Knows its own fitness. |
| `Population` | `eso/ga/population.py` | A generation of chromosomes. Trains, scores, replaces. |
| `SelectionOperator` | `eso/ga/selection.py` | Tournament selection of parents. |
| `GeneticOperator` | `eso/ga/operator.py` | Reproduction, mutation, crossover. |

## `eso.ga.gene`

The atom of the algorithm. A `Gene` carries no weights — it is a pointer to a strip of the mel-spectrogram. Position and height are expressed as integer indices on the frequency axis (`0` to `spec_height`). The constructor supports four modes: fully random, fixed position and free height, fixed height and free position, or fully fixed.

::: eso.ga.gene
    options:
      show_root_heading: false

## `eso.ga.chromosome`

The candidate solution. A `Chromosome` is an ordered list of genes plus the CNN trained on the bands those genes describe. It exposes `train()`, `get_fitness()`, `get_metric()`, and `get_genes()`. Fitness is computed relative to the baseline F1 and parameter count, weighted by `lambda_1` and `lambda_2` from [`ChromosomeConfig`](../configuration.md#chromosome).

::: eso.ga.chromosome
    options:
      show_root_heading: false

## `eso.ga.population`

A `Population` is a fixed-size set of chromosomes. It exposes methods to evaluate all chromosomes in a generation, locate the best, replace the weakest with offspring, and serialise itself to disk between generations.

::: eso.ga.population
    options:
      show_root_heading: false

## `eso.ga.selection`

Tournament selection. The only parameter is the tournament size `t`. Each call samples `t` chromosomes with replacement and returns the one with the highest fitness. Crossover requires two parents, so the operator is invoked twice for each crossover event.

::: eso.ga.selection
    options:
      show_root_heading: false

## `eso.ga.operator`

The genetic operators. `mutate` adds a bounded random delta to a gene's position, height, or both. `crossover` swaps a randomly chosen range of genes between two parents. `reproduce` copies a parent unchanged into the next generation. Rates are configured in [`GeneticOperatorConfig`](../configuration.md#genetic_operator) and must sum to one.

::: eso.ga.operator
    options:
      show_root_heading: false
