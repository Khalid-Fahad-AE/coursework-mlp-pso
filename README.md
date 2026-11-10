# MLP trained with Particle Swarm Optimisation

This project implements the coursework design:

- NumPy-only multilayer perceptron forward propagation.
- Generic PSO that knows nothing about neural networks.
- Adapter pattern connecting PSO to the MLP.
- Builder pattern for constructing the MLP.
- Strategy pattern for random and nearest-neighbour informants.
- Iris multiclass experiment, metrics, convergence plot, and comparison table.

## 1. Setup

```bash
cd coursework_mlp_pso
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The script writes:

- `results/experiment_results.csv`
- `results/convergence.png`

## 2. Dataset

The included example uses Iris from scikit-learn because it is small, reproducible,
multiclass, and requires no separate download. The MLP itself is still implemented
from scratch with NumPy.

To use a course CSV, replace `load_dataset()` in `data.py` so it returns:

```python
X: shape (number_of_rows, number_of_features)
y: shape (number_of_rows,) with integer class labels 0, 1, ...
```

Keep the train/test split and scaler fitting in `main.py`; never fit preprocessing
on the test set.

## 3. Coursework design mapping

| Requirement | Implementation |
|---|---|
| Activation hierarchy | `mlp.py` |
| Loss hierarchy | `losses.py` |
| Layer and Network | `mlp.py` |
| Builder pattern | `MLPBuilder` in `mlp.py` |
| Generic FitnessFunction | `pso.py` |
| Particle and Population | `pso.py` |
| Informant Strategy | `pso.py` |
| Adapter pattern | `adapter.py` |
| Main experiment | `main.py` |

## 4. Important parameter convention

For each layer, parameters are flattened in C order as:

1. all weights;
2. all biases.

Layers are concatenated from input to output. `get_parameters()` and
`set_parameters()` use exactly the same convention.

## 5. Report evidence

Use the printed results and generated files in the report. Explain:

1. the architecture and parameter-vector convention;
2. why PSO is independent from the MLP;
3. how the adapter supplies the fitness;
4. the effect of random versus nearest-neighbour informants;
5. convergence and final test metrics;
6. limitations, such as PSO requiring many fitness evaluations.
