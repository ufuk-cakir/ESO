# How ESO works

ESO frames band selection as an optimisation problem. The objective is twofold. Maximise the F1-score of a CNN classifier. Minimise the number of trainable parameters of that CNN. The search is run with a genetic algorithm.

The CNN architecture is identical to the baseline. Only the input shape changes. The baseline is trained on the full mel-spectrogram with low-pass filtering and downsampling. Each chromosome is evaluated against this baseline on the same validation set.

## Module map

| Stage | Module | Function |
| --- | --- | --- |
| Audio to spectrograms | [`eso.utils.preprocessing`](../api/utils.md) | Read audio files, parse annotations, segment to fixed-length windows, compute mel-spectrograms. |
| Baseline CNN | [`eso.model.model`](../api/model.md) | Train the reference CNN on the preprocessed dataset. |
| Population | [`eso.ga.population`](../api/ga.md) | Initialise N random chromosomes. |
| Per-chromosome training | [`eso.ga.chromosome`](../api/ga.md) | Train each chromosome's CNN on its extracted bands. |
| Selection and variation | [`eso.ga.selection`](../api/ga.md), [`eso.ga.operator`](../api/ga.md) | Tournament selection. Reproduction, mutation, crossover. |
| Final selection | [`eso.eso.ESO`](../api/eso.md) | Track best chromosome across all generations. |
| Evaluation | [`eso.utils.Evaluation`](../api/utils.md) | Sliding-window inference, calling-bout reconstruction, F1, FLOPs, RAM, energy. |

## Why an evolutionary algorithm

The search space of band positions and heights is large and discrete. The objective combines two non-aligned terms: classification quality and parameter count. The CNN's response to a given band set is non-differentiable with respect to the band geometry.

Evolutionary algorithms are nature-inspired methods for multi-objective optimisation that handle this setting well. They do not require gradients. They operate on a population, which reduces sensitivity to local optima. Constraints on the search space (fixed height, fixed position) are expressed as restricted initialisation. The paper uses tournament selection together with reproduction, mutation, and crossover.

## Fitness

Fitness is defined relative to the baseline.

\[
\text{Fitness}(c) \;=\; -\,\lambda_1 \frac{F1_{\text{baseline}} - F1_{c}}{F1_{\text{baseline}}} \;+\; \lambda_2 \frac{p_{\text{baseline}} - p_{c}}{p_{\text{baseline}}}
\]

The first term rewards classification quality. It is zero when the chromosome matches the baseline F1, negative when it underperforms, and positive when it surpasses it. The second term rewards parameter reduction. It is zero when the chromosome has as many parameters as the baseline, positive when it is smaller, and negative when it is larger.

Both terms are normalised by the baseline. The weights $\lambda_1$ and $\lambda_2$ therefore operate on comparable scales. The paper uses $\lambda_1 = 0.95, \lambda_2 = 0.05$ for the Thyolo Alethe and Pin-tailed Whydah datasets, and $\lambda_1 = 0.99, \lambda_2 = 0.01$ for the Hainan gibbon.

## Continue

- [Genes and chromosomes](genes-and-chromosomes.md). Representation.
- [Evolution](evolution.md). Selection, mutation, crossover, reproduction.
- [Spectrogram preprocessing](preprocessing.md). The two datasets per species.
- [CNN training and fitness](cnn-training.md). Per-chromosome training procedure.
- [Evaluation](evaluation.md). Sliding-window inference and the reported metrics.
