import unittest

import numpy as np

from losses import CategoricalCrossEntropy
from mlp import MLPBuilder
from pso import FitnessFunction, NearestNeighbourInformantSelection, Population


class ConstantFitness(FitnessFunction):
    def evaluate(self, position: np.ndarray) -> float:
        return float(np.sum(position**2))


class ComponentTests(unittest.TestCase):
    def test_parameter_round_trip(self) -> None:
        x = np.zeros((4, 2))
        y = np.eye(3)[[0, 1, 2, 0]]
        network, _ = (
            MLPBuilder()
            .set_dataset(x, y)
            .set_configuration({
                "hidden_layers": [4],
                "activations": ["tanh"],
                "task": "multiclass_classification",
            })
            .build()
        )
        parameters = np.arange(network.count_parameters(), dtype=float)
        network.set_parameters(parameters)
        np.testing.assert_array_equal(network.get_parameters(), parameters)

    def test_softmax_loss_and_generic_pso(self) -> None:
        loss = CategoricalCrossEntropy()
        self.assertAlmostEqual(loss.compute(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])), 0.0)
        population = Population(
            ConstantFitness(), NearestNeighbourInformantSelection(),
            num_particles=4, num_informants=2, num_iterations=3, dimension=2, seed=1
        )
        best_position, best_fitness = population.optimise()
        self.assertEqual(best_position.shape, (2,))
        self.assertGreaterEqual(best_fitness, 0.0)


if __name__ == "__main__":
    unittest.main()
