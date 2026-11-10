# Particle Swarm Optimisation of a Multilayer Perceptron

**Student:** [Your name]  
**Programme:** [Your programme]  
**Module:** Biologically Inspired Computation  
**Date:** [Submission date]

## Abstract

This project implements a multilayer perceptron (MLP) from scratch using NumPy and
optimises its parameters using Particle Swarm Optimisation (PSO). The MLP and PSO
are independent components connected by an adapter. The implementation also uses
the Builder pattern for network construction and the Strategy pattern for
interchangeable informant selection.

The UCI Banknote Authentication dataset was used for binary classification. The
main comparison used ten random seeds for both random and nearest-neighbour
informant selection. Random informants achieved mean test accuracy of
`0.8146 +/- 0.0843` and mean macro-F1 of `0.7939 +/- 0.1118`. Nearest-neighbour
informants achieved a slightly lower mean training loss, but lower mean test
accuracy (`0.7942 +/- 0.1049`) and macro-F1 (`0.7618 +/- 0.1623`). A separate
three-seed sensitivity experiment showed that a wider MLP produced the strongest
test performance among the tested configurations.

## 1. Introduction

The objective was to implement a generic PSO that can optimise an MLP without
containing MLP-specific code. This separation makes the optimiser reusable for
other models or mathematical functions.

The application performs the following steps:

1. Load the Banknote dataset.
2. Create stratified training, validation, and test sets.
3. Fit preprocessing on the training set only.
4. Build the MLP from a configuration.
5. Adapt the MLP to the generic PSO fitness interface.
6. Optimise the MLP parameters using PSO.
7. Evaluate the final parameter vector on validation and test data.
8. Compare informant strategies and hyperparameters.

## 2. Dataset and preprocessing

The Banknote Authentication dataset contains 1,372 observations, four continuous
features, and a binary target indicating authentic or forged banknotes. A
stratified 70/15/15 split was used:

| Subset | Rows |
|---|---:|
| Training | 960 |
| Validation | 206 |
| Test | 206 |
| Total | 1,372 |

Standardisation was fitted using the training subset only. The same fitted
transformation was then applied to validation and test data. This prevents
information leakage. The validation set was used for reporting and model
comparison; the test set was kept separate from PSO optimisation.

## 3. MLP design

The final binary-classification architecture is:

```text
4 input features -> 8 tanh neurons -> 5 sigmoid neurons -> 1 sigmoid neuron
```

The output neuron produces a probability between zero and one. Binary
cross-entropy is used as the loss function. The approximate random-guessing
cross-entropy baseline is `0.693`; the observed optimised losses are below this
value.

### Parameter vector

Parameters are flattened layer by layer. For each layer, all weights are stored
first, followed by all biases:

```text
Layer 1: 4*8 + 8 = 40 parameters
Layer 2: 8*5 + 5 = 45 parameters
Layer 3: 5*1 + 1 = 6 parameters
Total:             91 parameters
```

`get_parameters()` and `set_parameters()` use exactly the same ordering, so every
PSO position represents one valid complete MLP.

## 4. Software architecture

### MLP component

The MLP component contains:

- `ActivationFunction`, `Tanh`, `Sigmoid`, `Softmax`, and `Linear`;
- `Loss`, `MSE`, `BinaryCrossEntropy`, and `CategoricalCrossEntropy`;
- `Layer`, which stores weights, biases, and an activation object;
- `Network`, which performs forward propagation and parameter-vector operations;
- `MLPBuilder`, which constructs a valid architecture from configuration.

The MLP does not implement gradient descent or PSO.

### PSO component

The generic PSO component contains:

- `FitnessFunction`;
- `Particle`;
- `Population`;
- `InformantSelectionStrategy`;
- `RandomInformantSelection`;
- `NearestNeighbourInformantSelection`.

PSO operates only on real-valued vectors and scalar fitness values. It does not
know that the vectors represent neural-network weights.

### Adapter pattern

`MLPFitnessAdapter` implements `FitnessFunction`. For every candidate position it:

1. installs the position in the network;
2. performs a forward pass on all 960 training rows;
3. computes binary cross-entropy;
4. returns one scalar training loss.

Every fitness evaluation uses the complete training set, not a random minibatch.
Validation and test data are never passed to the adapter.

### Builder and Strategy patterns

The Builder pattern constructs the network from hidden-layer sizes, activation
names, and task type. The Strategy pattern allows random and nearest-neighbour
informants to be exchanged without changing `Particle` or `Population`.

## 5. Main PSO settings

| Parameter | Value |
|---|---:|
| Particles | 20 |
| Informants per particle | 3 |
| Iterations | 60 |
| Inertia weight | 0.7 |
| Cognitive coefficient | 1.5 |
| Social coefficient | 1.5 |
| Step size | 0.05 |
| Position limit | [-3, 3] |

The main experiment was repeated with seeds 0 through 9 for each informant
strategy. The reported values are means and sample standard deviations across
the ten runs.

## 6. Main ten-seed results

| Strategy | Training loss | Train accuracy | Validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|
| Random | 0.5588 +/- 0.0439 | 0.7811 +/- 0.0858 | 0.7956 +/- 0.0782 | 0.8146 +/- 0.0843 |
| Nearest-neighbour | 0.5549 +/- 0.0539 | 0.7667 +/- 0.1023 | 0.7791 +/- 0.1090 | 0.7942 +/- 0.1049 |

| Strategy | Test precision | Test recall | Test macro-F1 |
|---|---:|---:|---:|
| Random | 0.8540 +/- 0.0461 | 0.7986 +/- 0.0993 | 0.7939 +/- 0.1118 |
| Nearest-neighbour | 0.8435 +/- 0.0451 | 0.7787 +/- 0.1233 | 0.7618 +/- 0.1623 |

Nearest-neighbour informants produced a slightly lower mean training loss, but
random informants achieved better mean test accuracy, precision, recall, and F1.
Therefore, selecting the strategy by training loss alone would be misleading.
The standard deviations also show that results are sensitive to the random seed.

## 7. Hyperparameter sensitivity experiment

A separate experiment used three seeds per configuration and compared the baseline
with a smaller swarm, a longer PSO run, and a wider MLP.

| Configuration | Strategy | Test accuracy | Test macro-F1 |
|---|---|---:|---:|
| Baseline | Random | 0.8333 +/- 0.0691 | 0.8221 +/- 0.0780 |
| Baseline | Nearest | 0.8366 +/- 0.0112 | 0.8310 +/- 0.0080 |
| Smaller swarm | Random | 0.6262 +/- 0.0591 | 0.5069 +/- 0.1301 |
| Smaller swarm | Nearest | 0.7249 +/- 0.1339 | 0.6547 +/- 0.2274 |
| More iterations | Random | 0.8333 +/- 0.0691 | 0.8221 +/- 0.0780 |
| More iterations | Nearest | 0.8366 +/- 0.0112 | 0.8310 +/- 0.0080 |
| Wider MLP | Random | 0.8608 +/- 0.0587 | 0.8538 +/- 0.0652 |
| Wider MLP | Nearest | 0.8544 +/- 0.0825 | 0.8468 +/- 0.0912 |

The smaller swarm performed substantially worse, suggesting that population
diversity is important for this 91-dimensional search space. Increasing the
iteration budget from 60 to 100 produced almost identical results, indicating
that the baseline generally converged before the additional iterations. The wider
MLP achieved the strongest average test performance, although its variability
should be considered.

The complete precision, recall, loss, and standard-deviation values are stored in
`results/hyperparameter_summary.csv`.

## 8. Convergence analysis

The convergence plot shows the best-so-far training loss over PSO iterations. The
population stores the global best loss after initial evaluation and after each
iteration. Consequently, the plotted series is non-increasing; a later value
cannot be worse than an earlier global best.

The ten-seed plot should be interpreted using its mean curve and standard-deviation
band. A lower training curve does not necessarily imply better test performance,
as demonstrated by the main random-versus-nearest comparison.

## 9. Critical evaluation

### Strengths

- MLP forward propagation is implemented with NumPy.
- PSO is independent from MLP internals.
- Builder, Adapter, and Strategy patterns are explicit.
- Parameter flattening and reconstruction are consistent.
- Full-batch deterministic fitness evaluation is used.
- Training-only preprocessing prevents leakage.
- Both informant strategies and multiple seeds are evaluated.
- Hyperparameter sensitivity is reported.

### Limitations

- The main ten-seed experiment still has noticeable run-to-run variability.
- The hyperparameter sensitivity experiment uses three seeds per configuration,
  fewer than the main comparison.
- PSO requires many full forward-pass evaluations.
- Only a limited selection of architectures and PSO settings was tested.
- The test set is used for final comparison, so it should not be repeatedly used
  for tuning in a larger study.

### Future improvements

Future work could use nested validation, more seeds for every hyperparameter
configuration, confidence intervals, adaptive PSO coefficients, and comparison
with a gradient-based MLP baseline.

## 10. Conclusion

The coursework successfully implements a NumPy MLP whose parameters are optimised
by a generic PSO. The Adapter pattern preserves the separation between the two
components, while the Builder and Strategy patterns provide flexible construction
and informant selection.

Across ten seeds, random informants gave better average test generalisation even
though nearest-neighbour informants obtained slightly lower training loss. The
hyperparameter analysis showed that reducing swarm size harmed performance, extra
iterations beyond 60 gave little benefit, and a wider MLP produced the strongest
average test metrics among the tested configurations.

## Appendix: Reproduction commands

```bash
pip install -r requirements.txt
python -m unittest discover -s . -p 'test_*.py'
python run_experiments.py --data data_banknote_authentication.txt
python run_hyperparameter_experiments.py --data data_banknote_authentication.txt
```

Generated files:

```text
results/multi_seed_summary.csv
results/multi_seed_convergence.png
results/hyperparameter_summary.csv
```
