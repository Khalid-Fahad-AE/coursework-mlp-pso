# MLP trained with Particle Swarm Optimisation

This project implements the coursework design:

- NumPy-only multilayer perceptron forward propagation.
- Generic PSO that knows nothing about neural networks.
- Adapter pattern connecting PSO to the MLP.
- Builder pattern for constructing the MLP.
- Strategy pattern for random and nearest-neighbour informants.
- Banknote binary experiment, metrics, convergence plot, and comparison table.
- Iris remains available as a smoke-test demo when no dataset path is supplied.

## 1. Setup

```bash
cd coursework_mlp_pso
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --data data_banknote_authentication.txt
```

The script writes:

- `results/experiment_results.csv`
- `results/convergence.png`

## 2. Dataset

The final experiment expects the UCI Banknote Authentication file, usually named
`data_banknote_authentication.txt`. It must contain four feature columns followed by
the binary label, with no header. The MLP itself is implemented from scratch with
NumPy.

The script uses a stratified 70/15/15 train/validation/test split. The scaler is fit
on the training subset only. PSO evaluates the full training subset on every fitness
call; validation and test data are never passed to the adapter.

Run the 10-seed comparison with:

```bash
python run_experiments.py --data data_banknote_authentication.txt
```

This writes `multi_seed_results.csv`, `multi_seed_summary.csv`, and
`multi_seed_convergence.png` into `results/`.

For a quick smoke test, omit `--data`; the script uses Iris from scikit-learn.
To use a differently formatted course CSV, update `load_banknote_dataset()` in
`data.py` so it returns:

```python
X: shape (number_of_rows, number_of_features)
y: shape (number_of_rows,) with integer class labels 0, 1, ...
```

Keep the split and scaler fitting in `main.py`; never fit preprocessing on validation
or test data.

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
