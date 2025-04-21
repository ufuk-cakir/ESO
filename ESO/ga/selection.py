from .population import Population
import random
import numpy as np


class SelectionOperator:
    def __init__(self, tournament_size):
        """
        Initialize the SelectionOperator with a tournament size.
        """
        self.tournament_size = tournament_size

    def select_parents(self, population: Population) -> list:
        parent1 = self._select_one_parent(population)
        parent2 = self._select_one_parent(population)

        # Ensure that parent1 and parent2 are distinct individuals.
        while parent1 == parent2:
            parent2 = self._select_one_parent(population)

        return parent1, parent2

    def _select_one_parent(self, population: Population):
        """
        Select a parent individual from the population using tournament selection.

        :param population: The population from which to select a parent.
        :return: The best individual selected through tournament selection.
        """
        if not isinstance(population, Population):
            raise ValueError("The 'population' argument must be a Population object")

        if not population:
            raise ValueError("Population is empty")

        chromosomes = population.chromosomes

        # Randomly select `self.tournament_size` individuals from the population.
        ids = np.random.choice(chromosomes, size=self.tournament_size, replace=True)

        # Find the individual with the highest fitness.
        best = max(ids, key=lambda x: x.get_fitness())

        return best

    def best_individual(self, population, subset_size):
        """
        Select the best individual from a random subset of the population.

        :param population: The population from which to select the subset and the best individual.
        :param subset_size: The size of the random subset to select.
        :return: The best individual selected through tournament selection within the subset.
        """

        chromosomes = population.chromosomes

        if not population:
            raise ValueError("Population is empty")
        if subset_size > len(chromosomes):
            raise ValueError("Subset size is larger than the population size")

        # Randomly select `subset_size` individuals from the population
        subset = random.sample(chromosomes, subset_size)

        # initial the best individual in the random chosen subset
        best = None

        for ind in subset:
            ind.evaluate()
            if best is None or ind.fitness > best.fitness:
                best = ind

        # Return the best individual selected through tournament selection.
        return best
