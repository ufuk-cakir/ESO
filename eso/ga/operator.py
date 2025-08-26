#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:38:46 2023

@author: aaron-joel
"""
import copy

import random

from .chromosome import Chromosome
from .population import Population


# TODO add the mutation_position_range to hyperparameters
class GeneticOperator:
    def __init__(
        self,
        band_height_fixed,
        band_position_fixed,
        spec_height,
        crossover_rate,
        mutation_rate,
        reproduction_rate,
        mutation_height_range,
        mutation_position_range,
    ) -> None:
        """
        Initialize the GeneticOperator class.

        Parameters
        ----------
        config : dict
            Configuration file for input setting.

        Returns
        -------
        None
            DESCRIPTION.

        """
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._reproduction_rate = reproduction_rate
        self._mutation_height_range = mutation_height_range
        self._mutation_position_range = mutation_position_range
        self.spec_height=spec_height
        self._check_rates()
        self.band_height_fixed = band_height_fixed
        self.band_position_fixed = band_position_fixed

    def _check_rates(self):
        if round(self._mutation_rate + self._crossover_rate + self._reproduction_rate,2) != 1:
            print(round(self._mutation_rate + self._crossover_rate + self._reproduction_rate,2))
            raise ValueError(
                "The sum of mutation_rate, crossover_rate and reproduction_rate must be 1"
            )

    def set_crossover_rate(self, rate: float) -> None:
        """
        Set the crossover rate to the specified value of 'rate'

        Parameters
        ----------
        rate : float
            Propotion of population that should be crossover.

        Returns
        -------
        None

        """
        if not isinstance(rate, float):
            raise TypeError("crossover rate should be of type 'float'")

        if rate < 0 or rate >= 1.0:
            raise ValueError("crossover rate should be between 0.0 and 1.0")

        self._crossover_rate = rate

    def set_mutation_rate(self, rate: float) -> None:
        """
        Set the mutation rate to the specified value of 'rate'

        Parameters
        ----------
        rate : float
            Proportion of population that should be mutated.

        Returns
        -------
        None

        """
        if not isinstance(rate, float):
            raise TypeError("mutation rate should be of type 'float'")

        if rate < 0 or rate >= 1.0:
            raise ValueError("mutation rate should be between 0.0 and 1.0")

        self._mutation_rate = rate

    def get_crossover_rate(self) -> float:
        """
        Get the crossover rate

        Returns
        -------
        float
            The crossover rate.

        """
        return self._crossover_rate

    def get_mutation_rate(self) -> float:
        """
        Get the mutation rate

        Returns
        -------
        float
            The mutation rate.

        """
        return self._mutation_rate

    def random_crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple:
        """
        Takes two individuals from the population and generates two offsprings
        by exchanging a random number of genes at randomly selected position
        in both individual chromosomes.

        Parameters
        ----------
        parent1 : Chromosome
            The first parent.
        parent2 : Chromosome
            The second parent.

        Returns
        -------
        tuple
            A tuple of offsprings (Chromosome, Chromosome).

        """

        # Make a copy of the parents.
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)

        # Number of genes (No assumption is made about the chromosome length)
        n1 = len(parent1)
        n2 = len(parent2)
        n = min(n1, n2)

        # Randomly figure out how many genes to exchange
        m = random.randint(1, n)

        # Randomly select positions at which to exchange the genes for each
        # individual chromosome
        parent1_positions = random.sample(range(n1), m)
        parent2_positions = random.sample(range(n2), m)

        parent1_genes = parent1.get_genes()
        parent2_genes = parent2.get_genes()

        # Loop over the position in both chromosomes and exchange the genes
        for i, j in zip(parent1_positions, parent2_positions):
            # Exchange the ith gene of parent1 with the jth gene of parent2
            offspring1.set_gene(
                i,
                parent2_genes[j].get_band_position(),
                parent2_genes[j].get_band_height(),
            )
            offspring2.set_gene(
                j,
                parent1_genes[i].get_band_position(),
                parent1_genes[i].get_band_height(),
            )

        return (offspring1, offspring2)

    def crossover(self, parent1: Chromosome, parent2: Chromosome):
        """
        Takes two individuals from the population and generates two offsprings.

        Parameters
        ----------
        parent1 : Chromosome
            The first parent.
        parent2 : Chromosome
            The second parent.

        Returns
        -------
        tuple
            A tuple of offsprings (Chromosome, Chromosome).

        """

        # Make a deepcopy of the parents
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)

        # Number of genes
        # (Assumption: len(parent1) are not always the same len(parent2))
        n1 = len(parent1)
        n2 = len(parent2)
        n = min(n1, n2)

        # Perform the crossover operation

        # Pick a random number n, between 0 and n-1 and swap the genes
        # of parent1 and parent2 from [n:]

        # offspring1 = parent1[:n] + parent2[n:]
        # offspring2 = parent2[:n] + parent1[n:]

        start_idx = random.randint(0, n - 1)

        # modified the selection of end_idx instead of having it fixed
        end_idx = random.randint(start_idx + 1, n)

        # Let's ensure that the "whole" Chromosome is not changed.
        # This situation occurs when start_idx = 0 and end_idx = n
        if start_idx == 0 and end_idx == n:
            end_idx = n - 1

        parent1_genes = parent1.get_genes()
        parent2_genes = parent2.get_genes()

        for k in range(start_idx, end_idx):
            offspring1.set_gene(
                k,
                parent2_genes[k].get_band_position(),
                parent2_genes[k].get_band_height(),
            )
            offspring2.set_gene(
                k,
                parent1_genes[k].get_band_position(),
                parent1_genes[k].get_band_height(),
            )

        return (offspring1, offspring2)

    def _mutate_gene_position(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutate the gene by changing its band position.

        Parameters
        ----------
        chromosome : Chromosome
            The chromosome that should be mutated.

        Returns
        -------
        Chromosome
            The mutated chromosome.

        """

        # Get the length of the chromosome
        n = len(chromosome)

        genes = chromosome.get_genes()

        # Pick a random gene
        i = random.randint(0, n - 1)

        # Start with a deepcopy of the chromosome
        new_chromosome = copy.deepcopy(chromosome)

        # Adjust the position
        mutation_amount = random.randint(
            -self._mutation_position_range, self._mutation_position_range
            )

        new_position = max(
            genes[i].min_position,
            min(
                genes[i].get_band_position() + mutation_amount,
                genes[i].max_position ,
            ),
        )
        #ensure that the new parameter are constrained within the spectrogram
        if  new_position + genes[i].get_band_height() > self.spec_height : 
            new_position =  self.spec_height - genes[i].get_band_height() -  1



        # genes[i].set_band_position(new_position)
        new_chromosome.set_gene(position=i, band_position=new_position)
        return new_chromosome.sort()

    def _mutate_gene_height(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutate the gene by changing its band height.

        Parameters
        ----------
        chromosome : Chromsome
            The Chromosome that should be mutated.

        Returns
        -------
        Chromosome
            The mutated chromosome.

        """
        # Get the length of the chromosome
        n = len(chromosome)

        genes = chromosome.get_genes()

        # Pick a random gene
        i = random.randint(0, n - 1)

        # Start with a deepcopy of the chromosome
        new_chromosome = copy.deepcopy(chromosome)

        # Adjust the height slightly
        mutation_amount = random.randint(
            -self._mutation_height_range, self._mutation_height_range
        )  # Adjust this range as needed
        new_height = max(
            genes[i].min_height,
            min(genes[i].get_band_height() + mutation_amount, genes[i].max_height),
        )
        # genes[i].set_band_height(new_height)
        
        if genes[i].get_band_position() + new_height > self.spec_height : 
                new_height = self.spec_height - genes[i].get_band_position() - 1
             

        new_chromosome.set_gene(position=i, band_height=new_height)
        return new_chromosome

    def _mutate_gene(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutate the gene by changing its band position & height

        Parameters
        ----------
        chromosome : Chromosome
            The Chromosome that should be mutated.

        Returns
        -------
        Chromosome
            The mutated chromosome.

        """
        n = len(chromosome)
        genes = chromosome.get_genes()
        i = random.randint(0, n - 1)
        new_chromosome = copy.deepcopy(chromosome)

        # Apply Mutation on both

        height_mutation_amount = random.randint(
            -self._mutation_height_range, self._mutation_height_range
        )  # Adjust this range as needed
        pos_mutation_amount = random.randint(
            -self._mutation_position_range, self._mutation_position_range
        )
        
        new_height = max(
            genes[i].min_height,
            min(
                genes[i].get_band_height() + height_mutation_amount,
                genes[i].max_height,
            ),
        )
        new_position = max(
            genes[i].min_position,
            min(
                genes[i].get_band_position() + pos_mutation_amount,
                genes[i].max_position - genes[i].get_band_height(),
            ),
        )

        #ensure that the new parameter are constrained within the spectrogram
        #if the new band are outside of the spectrogram, keep the same height but change the position of the band
        if new_position + new_height > self.spec_height :
            new_position =  self.spec_height - new_height -  1

        
        new_chromosome.set_gene(i, new_position, new_height)

        return new_chromosome.sort()

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Randomly mutate the chromosome.

        Parameters
        ----------
        chromosome : Chromosome
            Chromosome to be mutated.

        Returns
        -------
        Chromosome
            The mutated Chromosome.

        """

        # First check if Chromosome has Stacked channels
        # Because then we dont want to change the height

        if chromosome._n_channels == 1:
            # Concatenated Chromosome: can mutate position and/or band height
            
            if not self.band_height_fixed  and not self.band_position_fixed : 
                # can mutate position and band height
                choice = random.randint(1, 3)

                if choice == 1:
                    mutated_chromosome = self._mutate_gene_position(chromosome)
                elif choice == 2:
                    mutated_chromosome = self._mutate_gene_height(chromosome)
                else:
                    mutated_chromosome = self._mutate_gene(chromosome)
            
            elif self.band_height_fixed  and not self.band_position_fixed : 
                # can mutate only position : 
                mutated_chromosome = self._mutate_gene_position(chromosome)

            elif not self.band_height_fixed  and self.band_position_fixed : 
                #can mutate only height :
                mutated_chromosome = self._mutate_gene_height(chromosome)

        else:
            # Stacked Chromosome: only mutate position
            mutated_chromosome = self._mutate_gene_position(chromosome)
        return mutated_chromosome

    def reproduce(self, population: Population) -> Chromosome:
        """
        Randomly selects an individual from the current population and
        moves it to the next generation.

        Parameters
        ----------
        population : Population
            The current population to select from.

        Returns
        -------
        Chromosome
            The randomly selected chromosome.

        """

        # Return a random chromosome from the population
        return random.choice(population.get_chromosomes())
