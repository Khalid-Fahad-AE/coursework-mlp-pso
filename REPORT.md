# Particle Swarm Optimisation of a Multilayer Perceptron

**Student:** [Your name]  
**Programme:** [Your programme]  
**Module:** Biologically Inspired Computation  
**Date:** [Submission date]

> **Draft status:** The numerical results in this document are from the Iris
> smoke-test run and must not be used as final Banknote coursework results. Run
> `python main.py --data data_banknote_authentication.txt` after adding the
> required Banknote file, then replace the results and discussion sections.

## Abstract

This project implements a multilayer perceptron (MLP) from scratch using NumPy and
optimises its weights and biases with Particle Swarm Optimisation (PSO). The MLP
and PSO are deliberately independent: PSO receives only a generic fitness-function
interface, while an adapter evaluates candidate MLP parameter vectors using the
training data. The implementation also demonstrates the Builder pattern for network
construction and the Strategy pattern for interchangeable informant selection.

The Iris dataset was used only as a reproducible three-class smoke-test example. Two
informant strategies were compared: random informants and Euclidean nearest-neighbour
informants. Nearest-neighbour PSO achieved the lower training loss (0.87080), while
random-informant PSO achieved better test performance, with accuracy 0.7667 and
macro-F1 0.7511. The convergence plot shows that nearest-neighbour selection improved
more rapidly and reached a lower training loss, but this did not translate into better
generalisation in this single run.

## 1. Introduction

The objective was to design a generic software system in which PSO can optimise an
MLP without containing neural-network-specific code. This separation is important
because the optimisation algorithm should be reusable for other models or mathematical
fitness functions.

The implemented system follows the required execution sequence:

1. Load and split a classification dataset.
2. Fit preprocessing on the training data only.
3. Construct an MLP from a configuration.
4. Adapt the MLP to the generic PSO fitness interface.
5. Optimise the complete MLP parameter vector using PSO.
6. Install the best parameters and evaluate on unseen test data.
7. Compare informant-selection strategies.

## 2. Dataset and preprocessing

The experiment uses the Iris dataset, which contains 150 observations, four numeric
features, and three classes. A stratified 80:20 split was used, giving 120 training
and 30 test observations. Stratification preserves the class proportions in both
subsets.

The four input features were standardised using `StandardScaler`. The scaler was
fitted only on the training set and then applied to both training and test sets. This
prevents information from the test set leaking into optimisation or preprocessing.
Targets were one-hot encoded for categorical cross-entropy.

## 3. Software architecture

### 3.1 MLP component

The MLP component contains:

- `ActivationFunction`, with `Tanh`, `Sigmoid`, `Softmax`, and `Linear`;
- `Loss`, with MSE, binary cross-entropy, and categorical cross-entropy;
- `Layer`, storing weights, biases, and an activation object;
- `Network`, providing forward propagation and parameter-vector operations;
- `MLPBuilder`, which constructs a valid architecture from a configuration.

The selected architecture is:

```text
4 input features -> 8 tanh neurons -> 5 sigmoid neurons -> 3 softmax neurons
```

The output layer has three neurons because the task is multiclass classification.
Categorical cross-entropy is used as the loss.

### 3.2 Parameter-vector convention

The network exposes all parameters as one vector. For every layer, the weights are
flattened first in NumPy C order, followed by the biases. Layers are concatenated
from input to output. For this architecture:

```text
Layer 1: 4*8 + 8 = 40 parameters
Layer 2: 8*5 + 5 = 45 parameters
Layer 3: 5*3 + 3 = 18 parameters
Total:             103 parameters
```

The same ordering is used by `get_parameters()` and `set_parameters()`, making the
PSO search-space representation unambiguous.

### 3.3 PSO component

PSO contains no MLP-specific imports or logic. Its main classes are:

- `FitnessFunction`: abstract evaluation interface;
- `Particle`: position, velocity, personal best, fitness, and informants;
- `Population`: initialisation, evaluation, best updates, and optimisation loop;
- `InformantSelectionStrategy`: interchangeable selection interface;
- `RandomInformantSelection`;
- `NearestNeighbourInformantSelection`.

The configured PSO values were:

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
| Random seed | 42 |

### 3.4 Adapter component

`MLPFitnessAdapter` implements `FitnessFunction`. For every candidate vector it:

1. installs the vector in the network;
2. performs a forward pass on `X_train`;
3. computes categorical cross-entropy against `y_train`;
4. returns the scalar training loss.

The adapter never receives test data, so test performance is not used by PSO.

## 4. Experimental method

The same MLP architecture, split, preprocessing, PSO settings, and seed were used
for both strategies. The independent variable was the informant-selection mechanism.
The dependent measurements were best training loss, test accuracy, macro-precision,
macro-recall, and macro-F1.

The current Iris run is only a software validation exercise, not the final
coursework experiment. The final Banknote study should repeat each configuration
over the requested seeds and report means and standard deviations.

## 5. Results

| Informant strategy | Best training loss | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Random | 0.89218 | 0.7667 | 0.8000 | 0.7667 | 0.7511 |
| Nearest-neighbour | 0.87080 | 0.6667 | 0.5000 | 0.6667 | 0.5556 |

Nearest-neighbour informants achieved a training loss approximately 2.4% lower than
random informants:

```text
(0.89218 - 0.87080) / 0.89218 * 100 = 2.4%
```

However, random informants produced the better test result: 10 percentage points
higher accuracy and approximately 0.1956 higher macro-F1. This indicates that the
lowest training loss was not the best indicator of test generalisation in this run.

## 6. Convergence analysis

The supplied convergence plot shows both strategies reducing the best-so-far training
loss over 60 iterations. Nearest-neighbour selection begins a strong decline earlier
and finishes around 0.8708. Random selection decreases more gradually and plateaus
around 0.8922.

The plot demonstrates convergence behaviour rather than proof of global optimality.
The curves are best-so-far values, so they should not increase. The difference between
training and test results also shows why the test set must remain separate from PSO.

## 7. Critical evaluation

### Strengths

- The MLP is implemented with NumPy rather than a neural-network framework.
- PSO is reusable because it depends only on `FitnessFunction`.
- Builder, Adapter, and Strategy patterns are explicit in the implementation.
- Parameter flattening and reconstruction use one consistent convention.
- Test data is excluded from optimisation.
- Both required informant strategies are compared.

### Limitations

- Results are from one stochastic run for each strategy.
- The experiment uses a small dataset and a modest PSO budget.
- PSO requires many full forward-pass evaluations and is computationally expensive.
- The current comparison changes only informant selection; it does not provide a
  complete grid search over all MLP and PSO hyperparameters.
- The selected hyperparameters were not statistically tuned.

### Improvements

Future work should repeat each strategy over multiple seeds, report mean and standard
deviation, test additional hidden-layer sizes, vary swarm size and iteration count,
and compare against a gradient-based baseline. A validation set could be used for
hyperparameter selection while preserving the test set for one final evaluation.

## 8. Conclusion for the smoke test

The project successfully implements a generic PSO and connects it to a from-scratch
MLP through an adapter. Both informant strategies reduce the training loss, with
nearest-neighbour informants obtaining the lower final training loss. Random informants
performed better on the held-out test set in the supplied run, showing that optimisation
of training loss and generalisation are related but not identical objectives.

The smoke-test result supports the architectural design: PSO can optimise the MLP without
understanding layers, activations, losses, or parameter storage.

These Iris numbers should be replaced before submission with results from the
Banknote dataset and the required multi-seed experiments.

## Appendix A: Reproduction commands

```bash
git clone https://github.com/Khalid-Fahad-AE/coursework-mlp-pso.git
cd coursework-mlp-pso
pip install -r requirements.txt
python -m unittest discover -s . -p 'test_*.py'
MPLBACKEND=Agg python main.py
```
