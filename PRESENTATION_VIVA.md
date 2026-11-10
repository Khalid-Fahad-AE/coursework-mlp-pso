# Presentation and Viva Preparation

## Suggested 10-slide presentation

### Slide 1 — Title and objective

**Title:** Particle Swarm Optimisation of a Multilayer Perceptron  
State that the objective is to implement an MLP from scratch and optimise its
parameters with a generic PSO.

### Slide 2 — Problem and design requirements

- MLP performs forward propagation and computes predictions.
- PSO minimises a generic scalar fitness.
- MLP and PSO must remain independent.
- Required patterns: Builder, Adapter, and Strategy.

### Slide 3 — Dataset and preprocessing

- Iris dataset: 150 samples, four features, three classes.
- Stratified 80:20 split.
- Standardisation fitted only on training data.
- One-hot targets for categorical cross-entropy.

### Slide 4 — MLP architecture

```text
4 -> 8 tanh -> 5 sigmoid -> 3 softmax
```

- 103 optimisable parameters.
- Softmax produces a probability distribution.
- Categorical cross-entropy evaluates multiclass predictions.

### Slide 5 — Parameter vector

Explain that every layer contributes flattened weights followed by biases. The
vector is concatenated layer by layer and reconstructed using the same sizes.

### Slide 6 — PSO design

- Each particle stores position, velocity, personal best, and informants.
- Population stores global best.
- Each iteration reassigns informants, updates velocities, moves particles,
  evaluates fitness, and updates best solutions.

### Slide 7 — Design patterns

- **Builder:** constructs the network from hidden-layer and task configuration.
- **Adapter:** converts MLP evaluation into `FitnessFunction.evaluate(position)`.
- **Strategy:** swaps random and nearest-neighbour informant selection without
  changing `Population`.

### Slide 8 — Experimental setup

Show the key settings: 20 particles, 3 informants, 60 iterations, inertia 0.7,
cognitive/social coefficients 1.5, step size 0.05, seed 42.

### Slide 9 — Results and convergence

| Strategy | Loss | Accuracy | F1 |
|---|---:|---:|---:|
| Random | 0.89218 | 0.7667 | 0.7511 |
| Nearest | 0.87080 | 0.6667 | 0.5556 |

Explain that nearest-neighbour converged to a lower training loss, but random
informants generalised better on the held-out test set.

### Slide 10 — Conclusion and future work

- Architecture met the independence requirement.
- Both strategies reduced loss.
- Random had better test performance in this run.
- Repeat across seeds and compare more hyperparameters in future work.

## Viva questions and model answers

### 1. Why use PSO instead of backpropagation?

PSO is a derivative-free population-based optimiser. It can optimise the network
without calculating gradients, which demonstrates a biologically inspired
optimisation method. The trade-off is that PSO generally needs many fitness
evaluations and may be less computationally efficient than backpropagation.

### 2. Why must PSO be independent of the MLP?

Independence gives reuse and testability. PSO only needs a vector position and an
`evaluate()` method, so the same optimiser can solve other numerical optimisation
problems without importing or understanding neural-network classes.

### 3. What is the role of the adapter?

The MLP does not naturally implement the PSO fitness interface. The adapter installs
a candidate vector, runs the MLP on training data, computes the loss, and returns a
scalar. It translates between the two component interfaces.

### 4. Why is the output activation softmax?

This is a three-class classification problem. Softmax converts the three output
scores into non-negative probabilities that sum to one, allowing the largest
probability to be selected as the predicted class.

### 5. Why use categorical cross-entropy?

The labels are one-hot encoded multiclass targets. Categorical cross-entropy
penalises assigning low probability to the correct class and is appropriate for
softmax probabilities.

### 6. Why is numerical clipping used in cross-entropy?

The logarithm of zero is undefined. Predictions are clipped to a small interval
before taking logarithms, preventing numerical errors and infinite loss values.

### 7. How many parameters does the network have?

There are 40 parameters in the 4-to-8 layer, 45 in the 8-to-5 layer, and 18 in
the 5-to-3 layer, giving 103 total parameters.

### 8. What does a particle represent?

A particle represents one complete candidate MLP parameter vector. Its position is
the current weights and biases, and its velocity controls how that vector changes.

### 9. What is the difference between personal best and global best?

Personal best is the best position discovered by one particle. Global best is the
best personal solution discovered by the entire population.

### 10. Why exclude a particle from its own informants?

The specification defines informants as other particles. Excluding the particle
prevents self-selection and ensures the neighbourhood actually provides social
information.

### 11. How are nearest neighbours selected?

Euclidean distance is calculated between current particle positions. The closest
`k` other particles are selected as informants.

### 12. Why did nearest-neighbour have lower training loss but worse test accuracy?

It found a better solution for the training objective in this run, but lower
training loss does not guarantee better generalisation. The result may also reflect
stochastic variation, the small test set, or the selected hyperparameters.

### 13. Is this evidence that random informants are always better?

No. It is evidence only for this dataset, architecture, seed, and PSO configuration.
Multiple independent runs are required before making a general claim.

### 14. How was data leakage avoided?

The split was performed before scaling. The scaler was fitted only on training data,
and the test set was used only after PSO had finished.

### 15. What would you improve?

Run multiple seeds, report confidence intervals or mean and standard deviation,
perform validation-based hyperparameter selection, test larger swarms and iteration
budgets, and compare with a gradient-descent baseline.

### 16. What happens if the parameter-vector ordering is inconsistent?

Weights and biases would be reconstructed into the wrong locations. The network
would then evaluate arbitrary parameters, causing incorrect fitness values and
preventing meaningful optimisation. That is why flattening and reconstruction use
the same documented convention.

### 17. Why is the test set not used inside the adapter?

Using test data during optimisation would leak evaluation information into training
and produce an overly optimistic estimate of generalisation.

### 18. What does convergence mean here?

The best-so-far training loss decreases and then changes only slightly. It indicates
that the swarm is making smaller improvements, but it does not prove that the global
minimum has been found.

## Presentation advice

- Explain the data flow clearly: `position -> adapter -> network -> loss`.
- Use the architecture diagram and parameter count to demonstrate implementation
  ownership.
- Do not claim nearest-neighbour is best overall; distinguish training loss from
  test performance.
- Explicitly state that the reported comparison is one stochastic run.
- If asked about limitations, volunteer the need for repeated seeds and uncertainty
  reporting.
