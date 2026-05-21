---
hide:
  - navigation
  - toc
---

<div class="eso-article-meta" markdown>
  <span class="eyebrow">Research Software</span>
  <span class="muted">Jeantet · Çakır · Lontsi · Dufourq</span>
  <span class="sep">·</span>
  <span class="muted">MIT licence</span>
</div>

<h1 class="eso-h1-large">Evolutionary Spectrogram <span class="accent">Optimisation</span></h1>

<p class="eso-deck">A genetic algorithm for selecting informative mel-frequency bands in passive acoustic monitoring. The selected bands form a reduced-size input to a CNN classifier whose architecture is otherwise unchanged from the baseline.</p>

<div class="eso-abstract">
  <div class="label">
    <span class="smallcaps">Abstract</span>
    <div class="meta">
      <div><b>Domain</b>Passive acoustic monitoring</div>
      <div><b>Targets</b>Hainan gibbon · Thyolo Alethe · Pin-tailed Whydah</div>
      <div><b>Hardware</b>Low-resource edge devices</div>
    </div>
  </div>
  <div class="body" markdown>
Convolutional neural networks for bioacoustic classification typically operate on full mel-spectrograms produced after low-pass filtering and downsampling. The networks are large and the preprocessing pipeline is computationally expensive, limiting deployment on resource-constrained devices.

This software implements ESO, an evolutionary algorithm that reduces the size of the network's input. A genetic algorithm searches over horizontal frequency bands of the unprocessed mel-spectrogram. Bands selected by the best chromosome are stacked or concatenated and used to train a CNN whose architecture is otherwise identical to the baseline. On the three datasets reported in the accompanying paper, ESO reduces mel-spectrogram size by 51 to 57 percent, trainable parameters by 64 to 72 percent, and energy consumption by 16 to 56 percent, while improving the F1-score by 1 to 6 percent.
  </div>
</div>

<h2 class="eso-numbered"><span class="n">1.</span> Datasets</h2>

ESO was evaluated on three publicly available bioacoustic datasets that differ in target species, recording rate, soundscape complexity, and call structure.

<figure markdown>
  ![Representative spectrograms from the three study datasets](assets/fig-datasets.png)
  <figcaption>Representative mel-spectrograms across the three study datasets. Panels show target vocalisations and the surrounding soundscape.</figcaption>
</figure>

<h2 class="eso-numbered"><span class="n">2.</span> Representation</h2>

A gene encodes a single mel-spectrogram band by its lower frequency boundary $P_k$ and height $h_k$, with $P_k \in [0, S_h - h_k]$, where $S_h$ is the mel-spectrogram height. A chromosome is an ordered collection of genes and constitutes one candidate solution.

<figure markdown>
  ![Representation of a chromosome with two genes](assets/fig-genes-chromosome.png)
  <figcaption>A chromosome with two genes applied to a mel-spectrogram of a Hainan gibbon vocalisation. Each gene defines a band by its position and height. Bands are extracted, then either stacked or concatenated.</figcaption>
</figure>

<h2 class="eso-numbered"><span class="n">3.</span> Core abstractions</h2>

Five classes cover the algorithm. Each lives under `eso/ga/` or `eso/model/` and is documented in the API reference.

<div class="eso-concepts" markdown>
  <div class="eso-concept" markdown>
### Gene
A single horizontal band, encoded as a position and a height.
<span class="path">eso.ga.gene.Gene</span>
  </div>
  <div class="eso-concept" markdown>
### Chromosome
An ordered set of genes and the CNN trained on the bands they define.
<span class="path">eso.ga.chromosome.Chromosome</span>
  </div>
  <div class="eso-concept" markdown>
### Population
A generation of chromosomes, evaluated and evolved together.
<span class="path">eso.ga.population.Population</span>
  </div>
  <div class="eso-concept" markdown>
### GeneticOperator
Reproduction, mutation, and crossover with user-defined rates.
<span class="path">eso.ga.operator.GeneticOperator</span>
  </div>
  <div class="eso-concept" markdown>
### SelectionOperator
Tournament selection. The tournament size is the only parameter.
<span class="path">eso.ga.selection.SelectionOperator</span>
  </div>
  <div class="eso-concept" markdown>
### Fitness
Relative F1 gain against the baseline plus relative reduction in trainable parameters.
<span class="path">Chromosome.get_fitness</span>
  </div>
</div>

<h2 class="eso-numbered"><span class="n">4.</span> Documentation</h2>

Installation, configuration, and the algorithm's details, with cross-references to the source code.

<div class="grid cards" markdown>

-   ### Installation

    System requirements, PyTorch wheel selection, and the editable install.

    [Read &rarr;](getting-started/installation.md)

-   ### First run

    Settings JSON template and a minimal Python API example.

    [Read &rarr;](getting-started/first-run.md)

-   ### How ESO works

    Pipeline, representation, evolution, training, and evaluation.

    [Read &rarr;](concepts/overview.md)

-   ### Configuration

    Every field of the settings JSON, with types, defaults, and recommended values from the paper.

    [Read &rarr;](configuration.md)

-   ### API reference

    Generated from docstrings in the `eso` package.

    [Read &rarr;](api/index.md)

-   ### Citation

    BibTeX and venue.

    [Read &rarr;](citation.md)

</div>
