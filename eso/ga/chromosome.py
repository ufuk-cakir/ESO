# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:17:35 2023

@author: ljeantet, ufuk-cakir
"""


import numpy as np
from .gene import Gene
from ..model.model import Model
import pickle
import os
from copy import deepcopy
import torch

# from .model.data import Data


class Chromosome:
    """Chromosome class

    This class represents a chromosome in the genetic algorithm and is a collection of genes.

    Parameters
    ----------
    num_genes : int
        The number of genes in the chromosome.
    min_num_genes : int
        The minimum number of genes in the chromosome.
    max_num_genes : int
        The maximum number of genes in the chromosome.
    baseline_metric : float
        The baseline metric to compare the accuracy to.
    baseline_parameters : int
        The baseline number of parameters to compare the number of trainable parameters to.
    gene_args : dict
        The arguments to pass to the gene class.
    model_args : dict
        The arguments to pass to the model class.
    lambda_1 : float
        The lambda_1 parameter for the fitness function.
    lambda_2 : float
        The lambda_2 parameter for the fitness function.
    logger : logging.Logger
        The logger to use.

    """

    def __init__(
        self,
        results_path,
        num_genes: int,
        min_num_genes: int,
        max_num_genes: int,
        baseline_metric: float,
        baseline_parameters: int,
        gene_args: dict,
        model_args: dict,
        architecture_args: dict,
        lambda_1: float = 0.5,
        lambda_2: float = 0.5,
        stack: bool = False,
        logger=None,
    ):
        self.results_path=results_path
        if num_genes== -1 : 
            self.num_genes = None 
        else :
            self.num_genes = num_genes
        self._min_num_genes = min_num_genes
        self._max_num_genes = max_num_genes
        self.logger = logger
        self._fitness = -np.inf  # Default to -inf
        self._metric = None
        self._metric_name = None
        self._trainable_parameters = None
        self._baseline_metric = baseline_metric
        self._baseline_parameters = baseline_parameters
        self.trained = False
        self._model_args = model_args
        self._architecture_args = architecture_args
        self._lambda_1 = lambda_1
        self._lambda_2 = lambda_2
        self._gene_args = gene_args
        self.stack = stack
        self._init_chromosome()


            

    def _init_chromosome(self):
        """Initialize the chromosome

        Initialize the chromosome by creating the genes.
        """

        # Check if min_num_genes is set
        if self.num_genes is None :
            if self._min_num_genes > self._max_num_genes:
                raise ValueError("min_num_genes cannot be greater than max_num_genes")
            # Randomly select the number of genes
            self.num_genes = np.random.randint(self._min_num_genes, self._max_num_genes)
        
        # Based on number of gene, adjust the minimum weight
        if not self.stack :
             self._gene_args['minimum_gene_height']=self._gene_args['minimum_gene_height']//self.num_genes+1
        
        # Initialize the genes
        self._genes = [Gene(**self._gene_args) for _ in range(self.num_genes)]
        self.sort()

    def sort(self):
        self._genes = sorted(self._genes, key=lambda x: x.get_band_position())
        return self

    def get_genes(self):
        """Get the genes in the chromosome

        Returns a list of the genes in the chromosome.

        Returns
        -------
        list
            The genes in the chromosome.
        """
        return self._genes

    def set_gene(self, position, band_position=None, band_height=None):
        """Set the gene at a specific position

        Set the gene at a specific position in the chromosome. Either the band position or the band height must be specified.
        This is used for crossover.

        Parameters
        ----------
        position : int
            The position of the gene to set.
        band_position : int, optional
            The position of the band to set.
        band_height : int, optional
            The height of the band to set.

        Raises
        ------
        ValueError
            If the position is greater than the number of genes.
        ValueError
            If the position is less than 0.
        ValueError
            If neither the band position or the band height is specified.
        """
        # do some checks
        # If we specificy both band_positions and band_height
        if position > self.num_genes:
            raise ValueError(
                f"Position {position} is greater than the number of genes {self.num_genes}"
            )
        if position < 0:
            raise ValueError(f"Position {position} is less than 0")
        if band_position is None and band_height is None:
            raise ValueError(
                "You need to specify either the band position or the band height"
            )

        if band_position is None and band_height is not None:
            band_position = self._genes[position].get_band_position()

        elif band_position is not None and band_height is None:
            band_height = self._genes[position].get_band_height()
        # Set the gene
        self._genes[position]._init_set_gene(band_position, band_height)
        self.sort()

    def get_info(self):
        """Get the information about the chromosome

        Returns a string with the information about the chromosome.
        """
        info = "Chromosome Info:\n"
        info += f"Number of Genes: {self.num_genes}\n"
        info += f"{self._metric_name.capitalize()}: {self._metric}\n"
        info += f"Trainable parameters: {self._trainable_parameters}\n"
        info += f"Fitness: {self._fitness}\n"
        info += "Genes: \n"
        for gene in self._genes:
            info += "\t" + str(gene) + "\n"
        return info

    def get_metric(self):
        """Get the accuracy/f1-score of the chromosome"""
        return self._metric

    def get_trainable_parameters(self):
        """Get the number of trainable parameters of the chromosome"""
        return self._trainable_parameters

    def train(self, X_train, Y_train, X_val, Y_val, save=False, model_name="eso_chromosome"):
        """Train the chromosome

        Create the sliced dataset from the encoded bands from the genes and train the model on this dataset.

        Parameters
        ----------
        X_train : np.array
            The training dataset.
        Y_train : np.array
            The training labels.
        X_val : np.array
            The validation dataset.
        Y_val : np.array
            The validation labels.
        """
        sliced_X_train = self._create_dataset(X_train)
        sliced_X_val = self._create_dataset(X_val)

        # And then train the model on this dataset
        image_shape = sliced_X_val.shape[1:]
        if len(image_shape) == 3:
            # this means the images have been stacked
            # TODO CHANGE This
            image_shape = image_shape[1:]
            # the first dimension is the number of channels, which is stored in
            # self._n_channels
        else:
            pass

        # Initialize Model
        model = Model(self.results_path,
            input_shape=(self._n_channels, image_shape[0], image_shape[1]),
            logger=self.logger, architecture_args=self._architecture_args,
            **self._model_args,
            use_chromosome=True,
        )

        # Train model
        self._loss = model.train(X_train=sliced_X_train, Y_train=Y_train, X_val=sliced_X_val, Y_val=Y_val, save=save, model_name=model_name)
        # Evaluate model validation accuracy/f1-score

        self._metric, self._metric_name = model.evaluate(
            X_val=sliced_X_val, Y_val=Y_val
        )
        self._trainable_parameters = model.get_number_of_parameters()
        self._fitness = self._calculate_fitness(
            self._metric, self._trainable_parameters
        )
        self.trained = True
        self._model_state_dict = deepcopy(model.get_model_dict())

    def get_fitness(self):
        """Get the fitness of the chromosome"""
        return self._fitness

    def save(self, path, name):
        """
        Save the chromosome to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the chromosome should be saved.
        name : str
            The name of the file.

        """
        # Create path name and add extension
        save_path = os.path.join(path, name + ".pkl")
        # Save as pickle
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    def save_model(self, path, name="chromosome_cnn_state"):
        save_path = os.path.join(path, name + ".pth")
        torch.save(self._model_state_dict, save_path)
        self.logger.info(f"CNN model state dict saved to {save_path}!")

    def predict(self, X, device, batch_size=128, threshold=0.5):
        bands = self._create_dataset(X)
        model = Model.load_cnn(self._model_state_dict, device)
        model.batch_size = batch_size  # TODO change this
        model.eval()
        prediction = model(bands)
        # Predict true label if prob([0,1]) > threshold
        prediction = (prediction[:, 1] > threshold).float()
        return prediction

    #this function doesn't work
    def evaluate(self, X, Y, device, batch_size=128, threshold=None):
        bands = self._create_dataset(X)
        
        model = Model(self.results_path,input_shape=(self._n_channels, bands.shape[1], bands.shape[2]), architecture_args=self._architecture_args, **self._model_args, use_chromosome=True)
        cnn = Model.load_cnn(self._model_state_dict, device)
        model._cnn = cnn
        model.batch_size = batch_size  # TODO change this
        metric, _ = model.evaluate(bands, Y, threshold)
        return metric

    def _calculate_fitness(self, metric, trainable_parameters):
        """Calculate the fitness of the chromosome"""
        fitness = (
            -self._lambda_1 * (self._baseline_metric - metric) / self._baseline_metric
            + self._lambda_2
            * (self._baseline_parameters - trainable_parameters)
            / self._baseline_parameters
        )
        return fitness

    def _get_bands(self, spectrogram):
        """Get the bands from the spectrogram

        Get the bands from the spectrogram based on the information contained in the genes.

        Parameters
        ----------
        spectrogram : np.array
            The spectrogram to extract the bands from.
        """
        # Read the amplitudes and sample rate
        bands = []
        band_heights = []
        for gene in self._genes:
            band_position = gene.get_band_position()
            band_height = gene.get_band_height()
            band = spectrogram[band_position : band_position + band_height, :]
            bands.append(band)
            band_heights.append(band_height)

        if np.all(np.array(band_heights) == band_heights[0]) and self.stack==True:
            # if all have same height --> stack images
            self._n_channels = len(bands)
            return np.stack(bands)
        else:
            self._n_channels = 1
            return np.concatenate(bands)

    def _create_dataset(self, dataset) -> np.array:
        """Create the sliced dataset

        Loop through the dataset and create the sliced dataset from the encoded bands from the genes.

        Parameters
        ----------
        dataset : np.array
            The dataset to create the sliced dataset from.
        """
        new_dataset = []
        for image in dataset:
            new_dataset.append(self._get_bands(image))

        return np.array(new_dataset)

    def __repr__(self):
        return f"Chromosome with {self.num_genes} genes"

    def __len__(self):
        return self.num_genes

    def __str__(self):
        gene_info = ""
        for i, gene in enumerate(self._genes):
            gene_info += f"Gene {i+1}: {gene}\n"

        return (
            f"Chromosome Info:\n"
            f"Number of Genes: {self.num_genes}\n"
            f"Validation {self._metric_name}: {self._metric}\n"
            f"Trainable parameters: {self._trainable_parameters}\n"
            f"Fitness: {self._fitness}\n"
            f"Genes: {gene_info}\n"
        )