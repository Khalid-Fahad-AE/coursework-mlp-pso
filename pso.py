"""Generic PSO component. No MLP-specific code belongs in this module."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FitnessFunction(ABC):
    @abstractmethod
    def evaluate(self, position: np.ndarray) -> float:
        raise NotImplementedError


class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.best_position = position.copy()
        self.fitness = np.inf
        self.best_fitness = np.inf
        self.informants: list[Particle] = []

    def evaluate(self, fitness_function: FitnessFunction) -> None:
        self.fitness = float(fitness_function.evaluate(self.position))

    def update_personal_best(self) -> None:
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

    def get_informants_best(self) -> np.ndarray:
        candidates = [self, *self.informants]
        return min(candidates, key=lambda particle: particle.best_fitness).best_position.copy()

    def update_velocity(
        self,
        informants_best: np.ndarray,
        global_best: np.ndarray,
        inertia_weight: float,
        cognitive_coefficient: float,
        social_coefficient: float,
        rng: np.random.Generator,
    ) -> None:
        r1 = rng.random(self.position.size)
        r2 = rng.random(self.position.size)
        self.velocity = (
            inertia_weight * self.velocity
            + cognitive_coefficient * r1 * (self.best_position - self.position)
            + social_coefficient * r2 * (informants_best - self.position)
            + social_coefficient * r2 * (global_best - self.position)
        )

    def update_position(self, step_size: float) -> None:
        self.position += step_size * self.velocity


class InformantSelectionStrategy(ABC):
    @abstractmethod
    def select_informants(
        self, particle: Particle, population: list[Particle], k: int
    ) -> list[Particle]:
        raise NotImplementedError


class RandomInformantSelection(InformantSelectionStrategy):
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def select_informants(self, particle: Particle, population: list[Particle], k: int) -> list[Particle]:
        candidates = [other for other in population if other is not particle]
        return list(self.rng.choice(candidates, size=k, replace=False))


class NearestNeighbourInformantSelection(InformantSelectionStrategy):
    def select_informants(self, particle: Particle, population: list[Particle], k: int) -> list[Particle]:
        candidates = [other for other in population if other is not particle]
        return sorted(candidates, key=lambda other: np.linalg.norm(particle.position - other.position))[:k]


class Population:
    def __init__(
        self,
        fitness_function: FitnessFunction,
        informant_strategy: InformantSelectionStrategy,
        num_particles: int,
        num_informants: int,
        num_iterations: int,
        dimension: int,
        inertia_weight: float = 0.7,
        cognitive_coefficient: float = 1.5,
        social_coefficient: float = 1.5,
        step_size: float = 0.05,
        position_limit: float = 3.0,
        seed: int = 42,
    ) -> None:
        if num_particles < 2:
            raise ValueError("At least two particles are required.")
        if not 1 <= num_informants <= num_particles - 1:
            raise ValueError("num_informants must satisfy 1 <= k <= N - 1.")
        if dimension < 1 or num_iterations < 1:
            raise ValueError("Dimension and iterations must be positive.")
        self.fitness_function = fitness_function
        self.informant_strategy = informant_strategy
        self.num_particles = num_particles
        self.num_informants = num_informants
        self.num_iterations = num_iterations
        self.dimension = dimension
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.step_size = step_size
        self.position_limit = position_limit
        self.rng = np.random.default_rng(seed)
        self.particles: list[Particle] = []
        self.global_best_position = np.zeros(dimension)
        self.global_best_fitness = np.inf
        self.history: list[float] = []

    def initialise(self) -> None:
        self.particles = [
            Particle(
                self.rng.uniform(-1.0, 1.0, self.dimension),
                self.rng.uniform(-0.1, 0.1, self.dimension),
            )
            for _ in range(self.num_particles)
        ]
        self.global_best_position = np.zeros(self.dimension)
        self.global_best_fitness = np.inf
        self.history = []

    def evaluate_particles(self) -> None:
        for particle in self.particles:
            particle.evaluate(self.fitness_function)

    def update_personal_bests(self) -> None:
        for particle in self.particles:
            particle.update_personal_best()

    def update_global_best(self) -> None:
        best = min(self.particles, key=lambda particle: particle.best_fitness)
        if best.best_fitness < self.global_best_fitness:
            self.global_best_fitness = best.best_fitness
            self.global_best_position = best.best_position.copy()

    def assign_informants(self) -> None:
        for particle in self.particles:
            particle.informants = self.informant_strategy.select_informants(
                particle, self.particles, self.num_informants
            )

    def optimise(self) -> tuple[np.ndarray, float]:
        self.initialise()
        self.evaluate_particles()
        self.update_personal_bests()
        self.update_global_best()
        self.history.append(self.global_best_fitness)
        for _ in range(self.num_iterations):
            self.assign_informants()
            for particle in self.particles:
                particle.update_velocity(
                    particle.get_informants_best(),
                    self.global_best_position,
                    self.inertia_weight,
                    self.cognitive_coefficient,
                    self.social_coefficient,
                    self.rng,
                )
                particle.update_position(self.step_size)
                particle.position = np.clip(
                    particle.position, -self.position_limit, self.position_limit
                )
            self.evaluate_particles()
            self.update_personal_bests()
            self.update_global_best()
            self.history.append(self.global_best_fitness)
        return self.get_best_solution()

    def get_best_solution(self) -> tuple[np.ndarray, float]:
        return self.global_best_position.copy(), float(self.global_best_fitness)
