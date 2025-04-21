import pickle
from tqdm import tqdm
from .chromosome import Chromosome
from ..model.data import Data


class Population:
    def __init__(
        self,
        pop_size: int,
        chromosome_args: dict,
        model_args: dict,
        gene_args: dict,
        logger=None,
        chromsomes: list = None,
        data: Data = None,
    ):
        """
        Initialize a population.
        """
        self.pop_size = pop_size
        # TODO CHANGE THIS TO DEFAULT
        self.logger = logger
        self._data = data
        self._chromosome_args = chromosome_args
        self._gene_args = gene_args
        self._model_args = model_args
        # self._set(config, logger, data)
        # Initialize Data
        # data_config = config["preprocessing"]
        # data_config["preprocess"] = False
        # data_config["force_recreate_dataset"] = False
        # This will create the dataset if it doesn't exist, and load it if it does
        # Pass this to the chromosome class so that it can use it to train and evaluate
        # self.data = Data(data_config, self.logger, verbose = False)

        if chromsomes is None:
            self._initial_population()
        else:
            self.chromosomes = chromsomes

    def reset_trained_flags(self):
        for chromosome in self.chromosomes:
            chromosome.trained = False

    def save(self, file_path, save_as_pickle=True):
        """
        Save the population to a file.

        :param path: The path to the file.
        :param save_as_pickle: Whether to save as a pickle file or not.
        """
        if save_as_pickle:
            self._save_population_to_pickle(file_path)
        else:
            raise NotImplementedError(
                "Only saving as pickle is supported at the moment."
            )

    def _save_population_to_pickle(self, file_path):
        # check if file path is directory or file
        # Check if file_path ends with .pkl
        if not file_path.endswith(".pkl"):
            file_path += ".pkl"
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path, logger, data):
        # Open the file for reading in binary mode ('rb')
        with open(file_path, "rb") as file:
            loaded_population = pickle.load(file)
        # loaded_population._set(config=config, logger=logger, data=data)
        loaded_population._data = data
        loaded_population.logger = logger
        return loaded_population

    def train_population(self, progress_bar_training=None, stop_event=None):
        """
        Train the population of chromosomes.

        :param epochs: The number of epochs to train each chromosome.
        """

        train_X, train_Y = self._data.get_data(type="train")
        val_X, val_Y = self._data.get_data(type="validation")
        # validation_loader = self._data.get_validation_data()
        if progress_bar_training is not None:
            progress_bar_training["maximum"] = self.pop_size
            progress_bar_training["value"] = 0

        for i, chromosome in enumerate(
            tqdm(
                self.chromosomes,
                desc="Training Chromosomes",
                total=len(self.chromosomes),
                ascii=" >=",
            )
        ):
            if stop_event is not None:
                if stop_event.is_set():
                    self.logger.info("Stopping training...")
                    return True
            if progress_bar_training is not None:
                progress_bar_training["value"] = i

            if chromosome.trained:
                self.logger.info(f"Skipping chromosome {i}.")
                continue
            else:
                # self.logger.info(f"Training chromosome {i} out of {self.pop_size}...")

                # chromosome._fitness = np.random.randint(0,1)
                chromosome.train(train_X, train_Y, val_X, val_Y)

        # After this each chrosomo will have a fitness value, calculated by the evaluate method in chromosome.py

    def get_chromosomes(self):
        """
        Get the chromosomes in the population.

        :return: The chromosomes in the population.
        """
        return self.chromosomes

    def replace_chromosomes(self, new_chromosomes):
        """
        Replace the chromosomes in the population with new ones.

        :param new_chromosomes: The new chromosomes to replace the old ones with.
        """

        self.chromosomes = new_chromosomes

    def get_best_chromosome(self) -> Chromosome:
        """
        Get the chromosome with the highest fitness value.

        Parameters
        ----------
        None

        Returns
        -------
        Chromosome
            The chromosome with the highest fitness value
        """
        # Check if the chromosomes have been evaluated
        if max(self.chromosomes, key=lambda x: x.get_fitness()) is None:
            raise ValueError("The chromosomes have not been evaluated yet.")
        return max(self.chromosomes, key=lambda x: x.get_fitness())

    def _initial_population(self):
        # Create the initial population of chromosomes based on the attributes of the class.
        self.chromosomes = [
            Chromosome(
                **self._chromosome_args,
                gene_args=self._gene_args,
                model_args=self._model_args,
                logger=self.logger,
            )
            for _ in range(self.pop_size)
        ]

    def __repr__(self) -> str:
        chromosome_repr = ", ".join(repr(chromosome) for chromosome in self.chromosomes)
        return f"Population(pop_size={self.pop_size}, chromosomes=[{chromosome_repr}])"

    def __str__(self) -> str:
        chromosome_str = "\n".join(str(chromosome) for chromosome in self.chromosomes)
        return f"Population Size: {self.pop_size}\nChromosomes:\n{chromosome_str}"

    def __len__(self):
        return len(self.chromosomes)
