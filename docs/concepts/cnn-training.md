# CNN training and fitness

Each chromosome owns a CNN that is trained on the bands selected by its genes. The architecture is the same as the baseline. Only the input shape differs.

## Baseline architecture

The baseline reproduces the simple CNN used in the prior literature on Hainan gibbon detection. It is intentionally small and is applied unchanged to all three datasets in the paper.

| Layer | Configuration |
| --- | --- |
| Convolution | 1 layer, 8 filters, kernel $8 \times 8$, ReLU |
| Max pooling | $4 \times 4$ |
| Flatten | (no parameters) |
| Fully connected | 32 units, ReLU |
| Output | 2 units, softmax |

Training settings, applied identically to the baseline and to per-chromosome CNNs:

| Setting | Value |
| --- | --- |
| Epochs | 30 |
| Batch size | 64 |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss | Cross-entropy |

These values are exposed as [`ModelConfig`](../configuration.md#model) and [`ArchitectureConfig`](../configuration.md#cnn_architecture) and can be changed for new datasets. The architecture is not optimised independently per species in the paper. The goal is to isolate the effect of ESO under comparable conditions.

## What `chromosome.train()` does

The procedure is the same as the baseline training, with two differences. The input is built from the bands defined by the chromosome's genes, and the spectrograms come from the unprocessed dataset rather than the filtered dataset.

1. Build the input tensor. For each clip, extract every band defined by the genes. Stack along a new depth axis or concatenate along the frequency axis.
2. Construct the CNN. Same architecture as the baseline, sized for the new input.
3. Train for 30 epochs with the settings above.
4. Evaluate on the validation set. F1 by default. This is `metric` in the fitness function.
5. Count trainable parameters. This is $p_{\text{chromosome}}$ in the fitness function.

## Input-shape validation

Gene heights can be small. A given convolution and pooling stack imposes a minimum legal input size. [`eso.model.cnn`](../api/model.md) walks the layer stack backwards to compute that minimum (`calc_back_conv`, `calc_back_pool`). ESO checks every gene against the minimum at startup. Invalid configurations fail before any training begins.

## Fitness

\[
\text{Fitness}(c) \;=\; -\,\lambda_1 \frac{F1_{\text{baseline}} - F1_{c}}{F1_{\text{baseline}}} \;+\; \lambda_2 \frac{p_{\text{baseline}} - p_{c}}{p_{\text{baseline}}}
\]

The first term is the relative improvement in F1-score over the baseline. The second term is the relative reduction in trainable parameters relative to the baseline.

A chromosome that matches the baseline F1 and has the same parameter count has fitness zero. A chromosome that improves F1 and reduces parameters has positive fitness. Both weights are non-negative. The framing is maximisation.

The paper uses:

| Dataset | $\lambda_1$ | $\lambda_2$ |
| --- | --- | --- |
| Hainan gibbon | 0.99 | 0.01 |
| Thyolo Alethe | 0.95 | 0.05 |
| Pin-tailed Whydah | 0.95 | 0.05 |

The Hainan gibbon weights are set tighter on F1 because the loss in detection accuracy is more sensitive on that dataset.

## Role of the baseline

The baseline serves three purposes.

1. Reference F1. The numerator of the F1 term in fitness.
2. Reference parameter count. The numerator of the parameter term in fitness.
3. Reference for the final comparison reported by [Evaluation](evaluation.md), including inference time, FLOPs, RAM, and energy.

The baseline is trained once at the start of an ESO run. Its F1 and parameter count are stored in `chromosome.baseline_metric` and `chromosome.baseline_parameters` and propagated to every chromosome.

## Final chromosome

The chromosome with the highest fitness across all generations is the ESO chromosome. Its CNN is the model that is evaluated against the baseline on the test set and used for any downstream deployment. There is no separate retraining stage in the current implementation.
